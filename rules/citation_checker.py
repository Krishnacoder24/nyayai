"""
checks citation spans against the IPC/BNS/Constitution corpus in Qdrant.
completely independent of the ML model — pure regex + vector retrieval.

works right now even without fine-tuned weights. only needs Qdrant running
and the corpus ingested (feature/corpus). if Qdrant is down, returns empty
list gracefully — never crashes the pipeline.

lookup itself goes through corpus.search.lookup_section rather than
querying Qdrant directly here — corpus.search is the single place that
knows the corpus's field names and collection, so rules code doesn't
have to stay in sync with it by hand.
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

# citation patterns found in Indian legal documents
# order matters — more specific patterns first
CITATION_PATTERNS = [
    # IPC sections: "Section 302 IPC", "Sec. 302 IPC", "S. 302 IPC"
    (r"[Ss]ec(?:tion|\.?)\.?\s*(\d+[A-Z]?)\s+IPC", "IPC"),
    # BNS sections: "Section 103 BNS"
    (r"[Ss]ec(?:tion|\.?)\.?\s*(\d+[A-Z]?)\s+BNS", "BNS"),
    # shorthand: "u/s 302 IPC", "u/s 103 BNS"
    (r"u/s\s+(\d+[A-Z]?)\s+(IPC|BNS)", None),  # act comes from capture group 2
    # Constitution articles: "Article 21", "Art. 21"
    (r"[Aa]rt(?:icle|\.?)\.?\s*(\d+[A-Z]?)\s+(?:of\s+the\s+)?Constitution", "Constitution"),
    # CrPC sections: "Section 144 CrPC"
    (r"[Ss]ec(?:tion|\.?)\.?\s*(\d+[A-Z]?)\s+CrPC", "CrPC"),
]


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
        logger.warning(f"qdrant not available ({e}) — skipping citation check")
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
            resolved_act = act if act else match.group(2)

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
    returns True if the section exists and is active in the corpus.
    returns False if not found or if found but marked repealed.
    """
    try:
        payload = lookup_section(section_no, act, client=client)

        if payload is None:
            return False

        # corpus ingestion (feature/corpus) should set status="active" or "repealed"
        return payload.get("status", "active") != "repealed"

    except Exception as e:
        logger.warning(f"qdrant query failed for Section {section_no} {act}: {e}")
        # if the query fails, don't flag it — false negatives are safer
        # than false positives for legal documents
        return True