"""
CPC parser. fully self-contained - no shared base class or inheritance
with any other act's parser (see issue #26). CPC (the Code of Civil
Procedure, 1908) is still active and unrepealed, and - unlike the
Constitution's Part -> Chapter -> Article structure - it follows the
same flat grammar as IPC/BNS/BNSS: a linear sequence of numbered
Sections, just grouped under "PART" instead of "CHAPTER".

page extraction, running-header stripping, and footnote-marker
resolution are shared plumbing, not act-specific grammar, so they still
come from corpus/pdf_utils.py rather than being reimplemented here -
same reasoning as BNS/BNSS.

what the real PDF actually looks like, verified against the real file
(corpus/sources/cpc/, "[As on the 10th January, 2026]" edition):

  - front matter (list of amending acts, abbreviations) is followed by
    a ~9-page "ARRANGEMENT OF SECTIONS" table of contents - same
    no-trailing-dash shape as IPC/BNS/BNSS, which is what makes the
    TOC-guided approach necessary here too.
  - sections run 1-158, with a handful of letter-suffixed insertions
    (21A, 35A, 35B, 44A, 87A, 87B, 99A, 100A, 111A, 135A, 148A, 153A,
    153B) and 13 stub entries ([Repealed.]/[Omitted.], e.g. 48, 66,
    68-72, 110, 111, 111A, 154-156) with no body text at all - same
    shape as IPC's repealed sections.
  - CAUTION: sections 157 ("Continuance of orders under repealed
    enactments") and 158 ("Reference to Code of Civil Procedure and
    other repealed enactments") are real, active sections with real
    body text - their TITLES just happen to contain the word
    "repealed". stub detection must check for a title that STARTS WITH
    "[Repealed"/"[Omitted" (same STUB_MARKERS convention as IPC/BNS/
    BNSS), never a bare substring match, or these two get wrongly
    discarded as empty stubs.
  - after section 158 the document continues into "THE FIRST SCHEDULE"
    - the Orders and Rules (procedural forms), which reuse their own
    per-Order numbering (Order I Rule 1, Order II Rule 1, ...) rather
    than continuing the Act's own section sequence. this is
    deliberately treated as out of scope for this parser (matches
    issue #29's framing of CPC as following the sibling acts' flat
    Chapter/Part -> Section grammar) - only the Act proper (sections
    1-158) is parsed. "THE FIRST SCHEDULE" appears exactly twice in the
    document: once as the TOC heading (used to bound toc_text) and once
    for the real Schedule body (used to bound body_text so section 158
    doesn't swallow the entire Schedule as its own body).
  - PART headers (PART I, PART II, ... PART XI) are the top-level
    grouping, same position/shape as IPC/BNS's CHAPTER headers just
    with a different keyword. below a PART, there are further unnumbered
    sub-headings (e.g. "PLACE OF SUING", "MISCELLANEOUS") that split a
    Part into named groups of sections - these aren't tracked in
    metadata, same as IPC/BNS not tracking sub-chapter headings either.
  - one real bug found and fixed at the shared-plumbing level while
    building this parser, not worked around here: corpus/pdf_utils.py
    used a single document-wide font-size baseline to decide whether a
    line was "smaller than body text" (and therefore the start of a
    footnote-definition block). ~30 of this PDF's 347 pages are
    genuinely typeset at a different body size than the document's
    median (e.g. section 60's page is set at 9pt against the document's
    11.04pt median, to fit its lettered sub-clauses) - a document-wide
    baseline misread an entire 9pt page as "smaller than normal" and
    swallowed real section text (60, 65, 67) into a phantom footnote
    block. fixed by comparing each page only against its own median
    instead - see pdf_utils.py's _page_font_size docstring for the full
    diagnosis. confirmed this didn't regress BNS/BNSS (still 358/358
    and 531/531 after the fix).

approach: same TOC-guided idea as IPC/BNS/BNSS - walk the TOC in order,
and for each expected (non-stub) number, search FORWARD from a
monotonically-advancing cursor for a candidate matching that exact
number. see ipc.py for the full rationale. verified end to end against
the real PDF: all 158 of 158 non-stub TOC entries find a body match, in
order, with sane boundaries.
"""

import re
from pathlib import Path

from corpus.schemas import Section
from corpus.pdf_utils import extract_pdf_pages, remove_repeated_headers

ACT = "CPC"
DEFAULT_STATUS = "active"  # CPC is still in force, unamended out of existence like IPC was

# confirmed via the Act's own commencement clause ("It shall come into
# force on the first day of January, 1909") - read directly out of the
# PDF, not assumed from memory, for the same reason BNS's effective date
# is read from its own footnote rather than guessed.
EFFECTIVE_DATE = "1909-01-01"

# marks where the TOC ends and the actual numbered Act text begins.
# CPC's enactment clause is phrased differently from BNS/BNSS's ("BE it
# enacted by Parliament") because CPC predates Indian independence and
# was enacted by the Governor-General in Council, not Parliament - this
# exact phrase is confirmed to occur exactly once in the whole document.
BODY_START_MARKER = "WHEREAS it is expedient to consolidate and amend the laws relating to the procedure"

# marks where the real section TOC starts. front matter before this
# (a "LIST OF AMENDING ACTS", itself a numbered "1. The Code of Civil
# Procedure (Amendment) Act, 1914..." list) also matches TOC_ENTRY's
# shape, so the TOC has to be bounded on both ends, not just the end -
# without this, that front-matter list gets parsed as if it were extra
# (garbage) sections.
TOC_START_MARKER = "ARRANGEMENT OF SECTIONS"

# after section 158 the PDF continues into the First Schedule (Orders
# and Rules), which is out of scope for this parser - see top docstring.
# "THE FIRST SCHEDULE" occurs exactly twice in the document: once as
# this TOC heading (used to bound the TOC) and once here (used to bound
# the body) - confirmed via a direct count against the real file.
TOC_END_MARKER = "THE FIRST SCHEDULE"
BODY_END_MARKER = "THE FIRST SCHEDULE"

# TOC entries: "9. Courts to try all civil suits unless barred.\n" - no
# dash, same shape as IPC/BNS/BNSS. unlike BNS, CPC's real TOC doesn't
# wrap titles across page breaks in a way that needed BNS's extra
# tolerance (verified directly: every one of CPC's 171 TOC entries
# extracts as a clean single line, including short ones like "Costs."
# and "Review." - no page-number-line debris bleeding into any title),
# so the simpler single-line IPC-style pattern is used as-is.
TOC_ENTRY = re.compile(r'\n\s*(\d{1,3}[A-Z]{0,2})\.\s+(.+)')

# PART headers, in both TOC and body: "PART I\nSUITS IN GENERAL". same
# position/shape as IPC/BNS's CHAPTER headers, just a different keyword
# - CPC groups its sections under numbered Parts, not Chapters. no
# letter-suffixed parts exist in this edition, but the optional footnote
# prefix is kept anyway for the same cheap-insurance reason BNS keeps it.
PART_START = re.compile(r'\n\s*(?:\{[^\}\n]*\}\s*\[)?\s*PART\s+([IVXLCDM]+[A-Z]?)\s*\n\s*([^\n]*)')

# section-start candidate, parameterised on the exact number currently
# expected from the TOC - same TOC-guided reasoning as IPC/BNS/BNSS (see
# ipc.py for why searching for a specific number beats a generic
# candidate scan).
#
# one difference from BNSS's template: the optional footnote-prefix's
# trailing "[" is made optional too (not just the whole prefix), because
# section 92's footnote is glued directly to the number with no
# following bracket at all ("{S. 92 shall not apply to any religious
# trust in Bihar...}92. Public charities.—...") - confirmed by running
# BNSS's stricter template first and checking what didn't match (157/158
# without this, 158/158 with it).
BODY_CANDIDATE_TEMPLATE = (
    r'(?:^|\n)\s*(?:\{{[^\}}\n]*\}}\s*\[?|\[)?\s*{number}(?![A-Za-z0-9])[\s.]{{1,3}}[-\u2013\u2014]?\s*'
    r'(?:[A-Za-z"\u2018\u201c][\s\S]{{0,250}}?)\.?\s*[-\u2013\u2014]'
)


def _candidate_pattern(number: str) -> re.Pattern:
    return re.compile(BODY_CANDIDATE_TEMPLATE.format(number=re.escape(number)), re.MULTILINE)


# TOC titles that mean "this section has no body text at all". checked
# with .startswith, never a substring test - sections 157/158 have real
# bodies but their titles happen to CONTAIN the word "repealed"
# ("Continuance of orders under repealed enactments"), so a substring
# check would wrongly treat them as stubs. see top docstring.
STUB_MARKERS = ("[omitted", "[repealed")


class CPCParser:
    act = ACT

    def parse(self, pdf_path: Path) -> list[Section]:
        raw_text = self._extract_raw_text(pdf_path)
        toc_text, body_text = self._split_toc_and_body(raw_text)
        toc_entries = self._parse_toc(toc_text)
        return self._parse_body(body_text, toc_entries)

    @staticmethod
    def _extract_raw_text(pdf_path: Path) -> str:
        # shared plumbing, same as BNS/BNSS - see corpus/pdf_utils.py
        pages = extract_pdf_pages(pdf_path)
        pages = remove_repeated_headers(pages)
        return "\n".join(pages)

    @staticmethod
    def _split_toc_and_body(raw_text: str) -> tuple[str, str]:
        toc_start_pos = raw_text.find(TOC_START_MARKER)
        if toc_start_pos == -1:
            raise ValueError(
                f"couldn't find '{TOC_START_MARKER}' - CPC PDF layout may have changed, "
                f"check where the table of contents begins"
            )

        toc_end_pos = raw_text.find(TOC_END_MARKER, toc_start_pos)
        if toc_end_pos == -1:
            raise ValueError(
                f"couldn't find '{TOC_END_MARKER}' - CPC PDF layout may have changed, "
                f"check where the table of contents ends"
            )
        toc_text = raw_text[toc_start_pos:toc_end_pos]

        marker_pos = raw_text.find(BODY_START_MARKER)
        if marker_pos == -1:
            raise ValueError(
                f"couldn't find '{BODY_START_MARKER}' - CPC PDF layout may have changed, "
                f"check the preamble/enactment clause wording"
            )
        body_text = raw_text[marker_pos:]

        # "THE FIRST SCHEDULE" occurs exactly twice - once as the TOC
        # heading (already consumed above) and once for real here.
        # searching from marker_pos (past the TOC) finds this second,
        # real occurrence, not the TOC heading again.
        end_pos = body_text.find(BODY_END_MARKER)
        if end_pos != -1:
            body_text = body_text[:end_pos]

        return toc_text, body_text

    @staticmethod
    def _parse_toc(toc_text: str) -> list[dict]:
        """returns ordered list of {"number", "title", "part", "is_stub"}."""
        parts = list(PART_START.finditer(toc_text))
        entries = []

        for match in TOC_ENTRY.finditer(toc_text):
            number = match.group(1)
            title = match.group(2).strip().rstrip(".")
            part = CPCParser._label_for_position(parts, match.start())
            is_stub = title.lower().startswith(STUB_MARKERS)
            entries.append({"number": number, "title": title, "part": part, "is_stub": is_stub})

        return entries

    @staticmethod
    def _parse_body(body_text: str, toc_entries: list[dict]) -> list[Section]:
        parts = list(PART_START.finditer(body_text))

        # same TOC-guided walk as IPC/BNS/BNSS: search forward from an
        # advancing cursor for the exact number currently expected,
        # never a generic "any number" scan - see ipc.py for the full
        # rationale on why this matters.
        matched: dict[int, re.Match] = {}
        cursor = 0
        for i, entry in enumerate(toc_entries):
            if entry["is_stub"]:
                continue
            pattern = _candidate_pattern(entry["number"])
            match = pattern.search(body_text, cursor)
            if match:
                matched[i] = match
                cursor = match.end()

        matched_positions = sorted(matched.items())
        sections = []

        for pos, (i, match) in enumerate(matched_positions):
            entry = toc_entries[i]
            body_start = match.end()
            body_end = matched_positions[pos + 1][1].start() if pos + 1 < len(matched_positions) else len(body_text)
            body = body_text[body_start:body_end]
            # the "N.—Title.—Body" shape (see BODY_CANDIDATE_TEMPLATE)
            # can leave a second, unconsumed dash right at the start of
            # what we capture as body - strip it rather than leave a
            # stray leading dash on the stored text.
            body = re.sub(r"^[\s\-\u2013\u2014]+", "", body).strip()
            # any section whose body happens to end right at a page
            # boundary picks up that page's bare footer number on its
            # own trailing line (e.g. section 35's body ends "...dated
            # (23-10-2020)].\n46") - remove_repeated_headers can't catch
            # these since each page number is different text, not a
            # repeated string. same issue confirmed and fixed in BNSS
            # (34 sections there); here it hits 3 (35, 119, 158).
            # section 158 additionally has a horizontal-rule footer
            # separator line ("______", confirmed 11 occurrences total
            # in the document) between the real content and the page
            # number - the optional group strips that too when present.
            # strips only a trailing STANDALONE line, never digits that
            # are part of the last real sentence, since those aren't on
            # their own line.
            body = re.sub(r"(\n_{3,})?\n\d{1,4}\s*$", "", body).strip()

            part = CPCParser._label_for_position(parts, match.start())
            metadata = {"part": part, "effective_date": EFFECTIVE_DATE}

            sections.append(Section(
                act=ACT, unit_type="section", number=entry["number"],
                title=entry["title"], body=body, status=DEFAULT_STATUS,
                metadata=metadata,
            ))

        # stub entries (Omitted/Repealed) never had body text to find -
        # record them factually from the TOC's own bracketed text,
        # nothing invented
        for entry in toc_entries:
            if entry["is_stub"]:
                metadata = {"part": entry["part"], "effective_date": EFFECTIVE_DATE}
                sections.append(Section(
                    act=ACT, unit_type="section", number=entry["number"],
                    title=entry["title"], body=f"[{entry['title']}]",
                    status=DEFAULT_STATUS, metadata=metadata,
                ))

        # sort by original TOC order for a predictable, sane output order
        order = {entry["number"]: i for i, entry in enumerate(toc_entries)}
        sections.sort(key=lambda s: order.get(s.number, len(order)))

        return sections

    @staticmethod
    def _label_for_position(headers: list[re.Match], position: int) -> str:
        """finds the Part header that comes right before this position in the text."""
        current = ""
        for header_match in headers:
            if header_match.start() > position:
                break
            number, title = header_match.group(1), header_match.group(2).strip()
            current = f"Part {number}: {title}" if title else f"Part {number}"
        return current