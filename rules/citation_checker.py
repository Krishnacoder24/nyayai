"""
checks citation spans against the IPC/BNS/BNSS/CPC/CrPC/Constitution corpus
in Qdrant. completely independent of the ML model - pure regex + vector
retrieval.

works right now even without fine-tuned weights. only needs Qdrant running
and the corpus ingested (feature/corpus). if Qdrant is down, returns empty
list gracefully - never crashes the pipeline.

lookup itself goes through corpus.search.lookup_section rather than
querying Qdrant directly here - corpus.search is the single place that
knows the corpus's field names and collection, so rules code doesn't
have to stay in sync with it by hand.

NOTE - CPC scope: cpc.py's parser only ingests the Act proper (Sections
1-158). The First Schedule (Order I Rule 1, Order XXI Rule 2, etc.) is
explicitly out of scope for that parser, so there's nothing in the corpus
to check an "Order X Rule Y CPC" citation against yet. Only Section-style
CPC citations are matched here on purpose - adding an Order/Rule pattern
now would just mean every such citation gets flagged "invalid" because
nothing in the corpus could ever confirm it, which is worse than not
checking it at all. Revisit if/when Order/Rule content gets ingested.
"""

import logging
import re

from qdrant_client import QdrantClient

from ocr.tokens import LineSpan
from model.schemas import ErrorSpan
from corpus.search import lookup_section

from config.settings import settings

logger = logging.getLogger(__name__)

QDRANT_URL = settings.qdrant_url

# acts that were wholesale superseded by a newer act (IPC -> BNS,
# CrPC -> BNSS, both effective 2024-07-01). ipc.py/crpc.py set
# status="repealed" on EVERY section of these two acts, regardless of
# whether that specific section is still validly cited - status alone
# can't mean "this citation is wrong" for these two acts, or nearly
# every real IPC/CrPC citation (which is most of what's actually filed
# in FIRs and pending prosecutions predating BNS/BNSS) would get
# flagged as an error. see _lookup_section's docstring.
SUPERSEDED_ACTS = {"ipc", "crpc"}

# citation patterns found in Indian legal documents
# order matters - more specific/longer act names first, so e.g. "BNSS"
# is matched before "BNS" gets a chance to (the trailing \b also guards
# against BNS matching as a prefix of BNSS on its own, but keeping the
# longer name first is the same defensive convention this file already
# used for BNS-before-IPC).
CITATION_PATTERNS = [
    # BNSS sections: "Section 103 BNSS"
    (r"[Ss]ec(?:tion|\.?)\.?\s*(\d{1,4}-?[A-Z]{0,2})\s+BNSS\b", "BNSS"),
    # BNS sections: "Section 103 BNS"
    (r"[Ss]ec(?:tion|\.?)\.?\s*(\d{1,4}-?[A-Z]{0,2})\s+BNS\b", "BNS"),
    # IPC sections: "Section 302 IPC", "Sec. 302 IPC", "S. 302 IPC"
    (r"[Ss]ec(?:tion|\.?)\.?\s*(\d{1,4}-?[A-Z]{0,2})\s+IPC\b", "IPC"),
    # CPC sections: "Section 80 CPC" - Section-only, see top docstring
    (r"[Ss]ec(?:tion|\.?)\.?\s*(\d{1,4}-?[A-Z]{0,2})\s+CPC\b", "CPC"),
    # CrPC sections: "Section 144 CrPC", also "Cr.P.C." / "Cr. P. C."
    # (both punctuated forms are common in real filings; the periods and
    # inner space are optional so plain "CrPC" still matches too)
    (r"[Ss]ec(?:tion|\.?)\.?\s*(\d{1,4}-?[A-Z]{0,2})\s+Cr\.?\s*P\.?\s*C\.?\b", "CrPC"),
    # shorthand: "u/s 302 IPC", "u/s 103 BNS", "u/s 103 BNSS", etc.
    # act comes from capture group 2 - BNSS listed before BNS for the
    # same reason as above.
    (r"u/s\s+(\d{1,4}-?[A-Z]{0,2})\s+(IPC|BNSS|BNS|CPC|Cr\.?\s*P\.?\s*C\.?)\b", None),
    # Constitution articles: "Article 21", "Art. 21"
    (r"[Aa]rt(?:icle|\.?)\.?\s*(\d{1,4}[A-Z]{0,2})\s+(?:of\s+the\s+)?Constitution", "Constitution"),
]

# maps whatever a matched CrPC variant looks like ("Cr.P.C", "Cr P C",
# "CrPC", ...) back to the exact act string crpc.py's ACT constant uses
# and corpus.search expects.
_ACT_CANONICAL = {"crpc": "CrPC"}


def _canonical_act(act: str) -> str:
    return _ACT_CANONICAL.get(re.sub(r"[.\s]", "", act).lower(), act)


def check_citations(spans: list[LineSpan]) -> list[ErrorSpan]:
    """
    extracts citation patterns from all spans, checks each against Qdrant,
    returns ErrorSpans for citations that don't match any known valid section.
    """
    try:
        client = QdrantClient(url=QDRANT_URL)
        # quick connectivity check before processing all spans
        client.get_collections()
    except Exception as e:
        logger.warning(f"qdrant not available ({e}) - skipping citation check")
        return []

    errors = []

    for span in spans:
        span_errors = _check_span(span, client)
        errors.extend(span_errors)

    return errors


def _check_span(span: LineSpan, client: QdrantClient) -> list[ErrorSpan]:
    errors = []

    for pattern, act in CITATION_PATTERNS:
        for match in re.finditer(pattern, span.text):
            section_no = match.group(1)

            # for u/s pattern, act comes from the match itself
            resolved_act = _canonical_act(act if act else match.group(2))

            is_valid = _lookup_section(client, section_no, resolved_act)

            if not is_valid:
                errors.append(ErrorSpan(
                    text=match.group(0),
                    error_type="citation",
                    page_no=span.page_no,
                    x0=span.x0, y0=span.y0, x1=span.x1, y1=span.y1,
                    suggestion=f"verify Section {section_no} {resolved_act} exists and is active",
                    confidence=0.95,  # regex match is deterministic, high confidence
                ))

    return errors


def _lookup_section(client: QdrantClient, section_no: str, act: str) -> bool:
    """
    returns True if the citation should be treated as valid, False if it
    should be flagged as an error.

    two different signals decide this, and they're not the same thing:

    - payload["title"] starting with "[Omitted"/"[Repealed" means THIS
      SPECIFIC SECTION never existed / was individually dropped before
      the act itself was ever superseded (e.g. IPC section 13, one of
      CrPC's own omitted sections). that's a real "doesn't exist" - flag it.

    - payload["status"] == "repealed" means something different for IPC
      and CrPC specifically: ipc.py/crpc.py stamp EVERY section with
      status="repealed" because BNS/BNSS replaced the whole act on
      2024-07-01 - not because that individual section is invalid. an
      FIR correctly citing "Section 302 IPC" for an offence from 2019 is
      not an error, and treating it as one would flag nearly every real
      IPC/CrPC citation. so `status` is ignored for IPC/CrPC and only
      the title-stub check applies; for BNS/BNSS/CPC/Constitution (whose
      `status` field is a genuine per-section signal, not an act-wide
      default) status is still checked as before.
    """
    try:
        payload = lookup_section(section_no, act, client=client)

        if payload is None:
            # CrPC stores letter-suffixed sections hyphenated exactly as
            # scanned from the TOC (e.g. "105-I"), so a document written
            # as "105I" or "105 I" would otherwise miss on formatting
            # alone, not because the section doesn't exist - retry once
            # with the hyphen toggled before giving up.
            if act.strip().lower() == "crpc":
                alt = _toggle_hyphen(section_no)
                if alt != section_no:
                    payload = lookup_section(alt, act, client=client)

        if payload is None:
            return False

        title = (payload.get("title") or "").strip().lower()
        if title.startswith(("[omitted", "[repealed")):
            return False

        if act.strip().lower() not in SUPERSEDED_ACTS and payload.get("status") == "repealed":
            return False

        return True

    except Exception as e:
        logger.warning(f"qdrant query failed for Section {section_no} {act}: {e}")
        # if the query fails, don't flag it - false negatives are safer
        # than false positives for legal documents
        return True


def _toggle_hyphen(number: str) -> str:
    """"105-I" <-> "105I" - see _lookup_section's docstring."""
    hyphenated = re.match(r"^(\d+)-([A-Za-z]+)$", number)
    if hyphenated:
        return hyphenated.group(1) + hyphenated.group(2)
    plain = re.match(r"^(\d+)([A-Za-z]{2,})$", number)
    if plain:
        return plain.group(1) + "-" + plain.group(2)
    return number