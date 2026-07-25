"""
CrPC parser. fully self-contained - no shared base class or inheritance
with any other act's parser (see issue #26). same reasoning as ipc.py:
CrPC is its own 50-year-old, heavily-amended act with its own real-world
PDF noise, and none of it is guaranteed to match IPC's grammar even
though the two acts look superficially similar (both pre-BNS/BNSS,
both TOC + numbered-section + footnote-marker shaped).

*** NOTE ON VERIFICATION LEVEL - READ BEFORE TRUSTING THIS FILE ***
The structural claims below were checked against the official CrPC text
at indiacode.nic.in (fetched directly, not recalled from training data)
on 2026-07-25. They were NOT checked against the specific PDF file that
will actually sit in corpus/sources/crpc/ - font sizes, footnote-marker
sizing, and exact TOC layout can only be confirmed the way ipc.py's own
comments do it: by running extract_pdf_pages() against the real file and
looking at the output. Treat this as a strong first draft, not a
"confirmed against the real file" claim.

confirmed differences from IPC's PDF (these are why this isn't a
straight copy of ipc.py with s/IPC/CrPC/):

  - no "WHEREAS it is expedient..." preamble. CrPC's enacting clause is
    "BE it enacted by Parliament in the twenty-fourth Year of the
    Republic of India as follows:-" - this is the primary body-start
    marker instead. "ACT NO. 2 OF 1974" kept as a fallback, same
    resilience pattern as IPC's fallback marker.
  - at least one hyphenated letter-suffixed section number exists:
    "105-I. Fine in lieu of forfeiture." (not "105I") - TOC/body number
    patterns allow an optional hyphen before the letter suffix.
  - at least one TOC entry has no space after the number's period:
    "144A.Power to prohibit carrying arms in procession..." - TOC_ENTRY
    uses a zero-or-more-whitespace match after the period, rather than
    IPC's mandatory-whitespace match.
  - "STATE AMENDMENT" blocks (Haryana/Karnataka/Maharashtra/etc.
    state-specific amendment text) are interleaved directly in the body
    between sections - NOT a footnote, NOT bracketed, just plain body
    text sitting between one section's end and the next section's
    start. currently these get folded into the preceding section's body
    text along with everything else between two matched section starts,
    same as any other body prose - they are NOT stripped out or
    special-cased. flagging this rather than silently deciding it's
    fine: if state-amendment text showing up inside a Section.body is a
    problem for downstream citation/entity checking, this needs a
    dedicated STATE_AMENDMENT_BLOCK pattern to excise it, likely
    similar in spirit to how footnote-definition blocks get excised.
  - chapters can have internal lettered sub-headings separate from the
    chapter itself, e.g. Chapter IV contains "A.-POWERS OF SUPERIOR
    OFFICERS OF POLICE" then later "B.-AID TO THE MAGISTRATES AND THE
    POLICE" as two sub-parts of the same chapter. this is DIFFERENT
    from IPC's "CHAPTER VA"-style lettered *chapter* suffix (which this
    parser also still supports, via CHAPTER_START, since CrPC has those
    too - e.g. "CHAPTER VIIA"). the A./B./C. sub-headings are NOT
    captured here - sections still only get labelled down to chapter
    granularity ("Chapter IV: ..."), same resolution IPC uses. If
    sub-chapter granularity turns out to matter, this needs its own
    pattern and its own field in metadata - not bolted onto
    _label_for_position's existing chapter-only logic.

what's assumed to carry over unchanged from IPC (untested against the
real file, but structurally identical in the indiacode.nic.in sample
checked - CrPC's footnote markers use the same superscript-digit +
bracket shape, e.g. "1[Provided that...]" with a footnote definition
"1. Ins. by Act 45 of 1978, s. 2 (w.e.f. 18-12-1978)." at the bottom of
the page):
  - footnote marker resolution (superscript-size detection, adjacency
    grouping, {curly brace} inline substitution)
  - TOC-guided body search (search forward for the exact next-expected
    number rather than trusting a generic "any number" scan)
  - stub (omitted/repealed) section handling via TOC title text
"""

import re
import statistics
from pathlib import Path

import pdfplumber
from pdfplumber.page import Page

from corpus.schemas import Section
from corpus.data.crpc_bnss_mapping import CRPC_TO_BNSS
from corpus.pdf_utils import remove_repeated_headers

ACT = "CrPC"
DEFAULT_STATUS = "repealed"  # BNSS replaced the CrPC in full, effective 2024-07-01
EFFECTIVE_DATE = "1974-04-01"  # Act No. 2 of 1974, commenced 1 April 1974 - confirmed via
                                # indiacode.nic.in / WIPO Lex, not assumed

# primary body-start marker: CrPC's enacting clause, textually adjacent
# to Chapter I / Section 1 (same "use the text right before the real
# body" strategy as IPC, just a different anchor phrase since CrPC has
# no WHEREAS preamble to key off of). confirmed present exactly once in
# the indiacode.nic.in copy checked; not yet confirmed against every
# possible edition.
ENACTING_CLAUSE_PATTERN = re.compile(
    r'BE\s+it\s+enacted\s+by\s+Parliament\s+in\s+the\s+twenty-fourth\s+Year',
    re.IGNORECASE,
)
# fallback marker, same role as IPC's BODY_START_MARKER_PATTERN
BODY_START_MARKER_PATTERN = re.compile(r'ACT\s*NO\.?\s*2\s*OF\s*1974', re.IGNORECASE)

# TOC lines: "   9.   Court of Session.\n" or "   105-I. Fine in lieu...\n"
# or "   144A.Power to prohibit...\n" (no space after the period, confirmed
# present in the real TOC) - no dash, ends at newline. the optional
# hyphen before the letter suffix (105-I) and the "\.\s*" rather than
# "\.\s+" are the two confirmed CrPC-specific relaxations vs IPC's
# TOC_ENTRY.
TOC_ENTRY = re.compile(r'\n\s*(\d{1,3}-?[A-Z]{0,2})\.\s*(.+)')

# chapter headers, in both TOC and body: "CHAPTER XVI" / "CHAPTER VIIA"
# (roman + optional letter, e.g. the real "CHAPTER VIIA" - "RECIPROCAL
# ARRANGEMENTS..." - confirmed in the TOC). same footnote+bracket
# tolerance as IPC in case an inserted chapter is glued to a marker the
# same way IPC's inserted chapters are (not yet confirmed CrPC actually
# does this to any of its chapter headers, but costs nothing to allow
# for and matches the same shape).
CHAPTER_START = re.compile(r'\n\s*(?:\{[^\}\n]*\}\s*\[)?\s*CHAPTER\s+([IVXLCDM]+[A-Z]?)\s*\n\s*([^\n]*)')

# template for a section-start candidate, parameterised on the EXACT
# number currently expected from the TOC (see _parse_body) - same
# TOC-guided strategy as IPC, same rationale: searching for a specific
# expected number makes footnote/bracket/STATE AMENDMENT noise harmless
# by construction, since a false-positive match is simply never the
# number being waited for.
BODY_CANDIDATE_TEMPLATE = (
    r'(?:^|\n)\s*(?:\{{[^\}}\n]*\}}\s*)?(\[)?\s*{number}(?![A-Za-z0-9])[\s.]{{1,3}}'
    r'(?:[A-Za-z"\u2018\u201c][\s\S]{{0,250}}?)\.\s*[-\u2013\u2014]'
)


def _candidate_pattern(number: str) -> re.Pattern:
    return re.compile(BODY_CANDIDATE_TEMPLATE.format(number=re.escape(number)), re.MULTILINE)


# --- footnote-marker resolution: unchanged from ipc.py's approach.
# duplicated rather than imported, same as ipc.py does relative to
# pdf_utils.py - keeps this act's parser fully self-contained so a
# future CrPC-specific quirk in its footnote layout can be fixed here
# without touching IPC (see corpus/parser.py's module docstring on
# issue #26 - independence is about not sharing a parsing class, not
# about avoiding plain PDF-reading plumbing, but footnote resolution
# specifically is treated as act-specific like IPC treats it, since
# it's tightly coupled to each act's own font-size quirks).
SUPERSCRIPT_SIZE_RATIO = 0.85
FOOTNOTE_ENTRY_START = re.compile(r'^\s*(\d{1,3})\.\s+\S')
FOOTNOTE_NUMBER_PREFIX = re.compile(r'^\s*\d{1,3}\.\s*')
FOOTNOTE_REGION_MAX_TOP_FRACTION = 0.5
MARKER_DIGIT_ADJACENCY_RATIO = 0.5
PAGE_NUMBER_LINE = re.compile(r'^\d{1,4}$')
TRAILING_PAGE_NUMBER = re.compile(r'(?<=[.\)])\s+\d{1,4}\s*$')


def _dominant_font_size(pdf: pdfplumber.PDF) -> float:
    sizes = [
        char["size"]
        for page in pdf.pages
        for char in page.chars
        if char.get("text", "").strip()
    ]
    return statistics.median(sizes) if sizes else 0.0


def _group_chars_into_lines(chars: list[dict]) -> list[list[dict]]:
    lines: dict[int, list[dict]] = {}
    for ch in chars:
        if ch.get("object_type") != "char":
            continue
        if not ch.get("text", "") or (not ch["text"].strip() and ch["text"] != " "):
            continue
        key = round(ch["top"])
        lines.setdefault(key, []).append(ch)
    return [lines[key] for key in sorted(lines)]


def _line_text(line: list[dict]) -> str:
    return "".join(c["text"] for c in sorted(line, key=lambda c: c["x0"]))


def _find_footnote_region_start(lines: list[list[dict]], page_height: float, baseline: float) -> int | None:
    """returns the index into `lines` where the footnote-definition block
    starts, or None if this page has no footnote block. a line only
    counts as the start of one if it (a) begins with the "N. " shape,
    (b) sits in the bottom portion of the page, and (c) is set smaller
    than body text.

    NOTE: ipc.py's version of this function adds a fourth check - the
    gap between this line and the previous line must be unusually large,
    to avoid mistaking a real numbered body clause for a footnote start.
    that check is deliberately NOT used here: confirmed against real
    CrPC output that it produces a false negative - a genuine footnote
    block (the "1. The words "except the State of Jammu and Kashmir"..."
    entry under section 1(2)) failed to be detected, leaving the
    definition text sitting uncut in the body and its marker falling
    back to the bare "{1}" placeholder instead of resolving. CrPC's
    footnote separator appears to be a drawn horizontal rule rather than
    extra line-spacing, which a text-position gap check has nothing to
    key off of. dropping the gap check re-opens the false-positive risk
    it was guarding against in IPC (a real numbered body clause in the
    bottom half of the page, at a genuinely smaller font, being
    mis-read as a footnote) - not yet confirmed whether that actually
    happens anywhere in CrPC's real text. if it turns out to, this needs
    a CrPC-specific way to detect the real separator (e.g. pdfplumber's
    page.lines / page.rects for the drawn rule) rather than reusing
    IPC's gap heuristic."""
    if baseline <= 0:
        return None

    for i, line in enumerate(lines):
        if not line:
            continue
        top = min(c["top"] for c in line)
        if top < page_height * FOOTNOTE_REGION_MAX_TOP_FRACTION:
            continue
        if not FOOTNOTE_ENTRY_START.match(_line_text(line)):
            continue
        avg_size = statistics.mean(c["size"] for c in line if c["text"].strip())
        if avg_size >= baseline * SUPERSCRIPT_SIZE_RATIO:
            continue
        return i

    return None


def _strip_trailing_page_number(text: str) -> str:
    return TRAILING_PAGE_NUMBER.sub("", text)


def _extract_footnote_definitions(lines: list[list[dict]], start_idx: int) -> dict[str, str]:
    entries: dict[str, str] = {}
    current_num: str | None = None
    current_parts: list[str] = []

    for line in lines[start_idx:]:
        text = _line_text(line).strip()
        if not text or PAGE_NUMBER_LINE.match(text):
            continue
        match = FOOTNOTE_ENTRY_START.match(text)
        if match:
            if current_num is not None:
                entries[current_num] = _strip_trailing_page_number(" ".join(current_parts).strip())
            current_num = match.group(1)
            current_parts = [FOOTNOTE_NUMBER_PREFIX.sub("", text, count=1)]
        elif current_num is not None:
            current_parts.append(text)

    if current_num is not None:
        entries[current_num] = _strip_trailing_page_number(" ".join(current_parts).strip())

    return entries


def _is_marker_digit(ch: dict, line: list[dict], baseline: float) -> bool:
    if not ch["text"].isdigit():
        return False
    if ch["size"] >= baseline * SUPERSCRIPT_SIZE_RATIO:
        return False

    neighbor_sizes = [c["size"] for c in line if c is not ch and not c["text"].isdigit() and c["text"].strip()]
    if not neighbor_sizes:
        return True

    local_size = statistics.median(neighbor_sizes)
    return ch["size"] < local_size * SUPERSCRIPT_SIZE_RATIO


def _resolve_markers_in_line(line: list[dict], baseline: float, footnotes: dict[str, str]) -> None:
    line_sorted = sorted(line, key=lambda c: c["x0"])
    i = 0
    while i < len(line_sorted):
        ch = line_sorted[i]
        if not _is_marker_digit(ch, line_sorted, baseline):
            i += 1
            continue

        run = [ch]
        j = i + 1
        while j < len(line_sorted):
            nxt = line_sorted[j]
            if not _is_marker_digit(nxt, line_sorted, baseline):
                break
            gap = nxt["x0"] - run[-1]["x1"]
            if gap >= run[-1]["size"] * MARKER_DIGIT_ADJACENCY_RATIO:
                break
            run.append(nxt)
            j += 1

        number = "".join(c["text"] for c in run)
        footnote_text = footnotes.get(number)
        # {footnote text} when resolved, falls back to bare {number} if
        # no definition was found on this page - never invents footnote
        # text that isn't there. curly braces (not square brackets) so a
        # resolved footnote is never confused with the Act's own real
        # "[...]" brackets around amended/inserted sections - the two
        # can end up sitting right next to each other:
        # "{footnote text}[inserted section text]".
        run[0]["text"] = f"{{{footnote_text}}}" if footnote_text else f"{{{number}}}"
        for extra in run[1:]:
            extra["text"] = ""

        i = j


def _resolve_footnote_markers(page: Page, baseline: float) -> Page:
    if baseline <= 0:
        return page

    lines = _group_chars_into_lines(page.chars)
    footnote_start_idx = _find_footnote_region_start(lines, page.height, baseline)

    footnotes: dict[str, str] = {}
    body_line_count = len(lines)
    if footnote_start_idx is not None:
        footnotes = _extract_footnote_definitions(lines, footnote_start_idx)
        body_line_count = footnote_start_idx

        for line in lines[footnote_start_idx:]:
            for ch in line:
                ch["text"] = ""

    for line in lines[:body_line_count]:
        _resolve_markers_in_line(line, baseline, footnotes)

    return page


def _drop_blank_lines(text: str) -> str:
    """removes lines that are empty or whitespace-only. blanking a
    footnote-definition line's characters (see _resolve_footnote_markers)
    only empties the characters, not the line's vertical space -
    pdfplumber's extract_text() still emits a blank line there, since it
    places line breaks by character position, not by whether any text
    survived. without this, every footnote block leaves a visible gap of
    empty lines behind exactly where it used to sit. same helper as
    pdf_utils.py's version - duplicated rather than imported since it's
    private there and this file follows the same self-contained
    convention as ipc.py."""
    return "\n".join(line for line in text.split("\n") if line.strip())


# TOC titles that mean "this section has no body text at all" - same
# generic, title-text-driven mechanism as IPC. not yet confirmed which
# (if any) CrPC sections are actually omitted/repealed in the TOC - the
# mechanism doesn't need that confirmed in advance, it reads whatever
# the real TOC says.
STUB_MARKERS = ("[omitted", "[repealed")


class CRPCParser:
    act = ACT

    def parse(self, pdf_path: Path) -> list[Section]:
        raw_text = self._extract_raw_text(pdf_path)
        toc_text, body_text = self._split_toc_and_body(raw_text)
        toc_entries = self._parse_toc(toc_text)
        return self._parse_body(body_text, toc_entries)

    @staticmethod
    def _extract_raw_text(pdf_path: Path) -> str:
        with pdfplumber.open(pdf_path) as pdf:
            baseline = _dominant_font_size(pdf)
            pages = [
                _drop_blank_lines(_resolve_footnote_markers(page, baseline).extract_text() or "")
                for page in pdf.pages
            ]
        pages = remove_repeated_headers(pages)
        return "\n".join(pages)

    @staticmethod
    def _split_toc_and_body(raw_text: str) -> tuple[str, str]:
        enacting_match = ENACTING_CLAUSE_PATTERN.search(raw_text)
        if enacting_match is not None:
            # back up to the nearest preceding "CHAPTER" heading, same
            # reasoning as IPC: keep Chapter I's real heading in
            # body_text where CHAPTER_START can find it, rather than
            # stranding it in toc_text.
            chapter_pos = raw_text.rfind("\nCHAPTER", 0, enacting_match.start())
            marker_pos = chapter_pos if chapter_pos != -1 else enacting_match.start()
            return raw_text[:marker_pos], raw_text[marker_pos:]

        # fallback: this edition doesn't phrase its enacting clause the
        # expected way - try the act-number line instead.
        matches = list(BODY_START_MARKER_PATTERN.finditer(raw_text))
        if not matches:
            raise ValueError(
                "couldn't find CrPC's body-start point - neither the enacting "
                "clause ('BE it enacted by Parliament in the twenty-fourth "
                "Year...') nor an 'ACT NO. 2 OF 1974'-style marker were found "
                "anywhere in the extracted text. CrPC PDF layout may differ "
                "from the indiacode.nic.in edition this parser was checked "
                "against - check the text right after the table of contents "
                "ends and update ENACTING_CLAUSE_PATTERN / "
                "BODY_START_MARKER_PATTERN."
            )
        marker_pos = matches[-1].start()
        return raw_text[:marker_pos], raw_text[marker_pos:]

    @staticmethod
    def _parse_toc(toc_text: str) -> list[dict]:
        """returns ordered list of {"number", "title", "chapter", "is_stub"}."""
        chapters = list(CHAPTER_START.finditer(toc_text))
        entries = []

        for match in TOC_ENTRY.finditer(toc_text):
            number = match.group(1)
            title = match.group(2).strip().rstrip(".")
            chapter = CRPCParser._label_for_position(chapters, match.start())
            is_stub = title.lower().startswith(STUB_MARKERS)
            entries.append({"number": number, "title": title, "chapter": chapter, "is_stub": is_stub})

        return entries

    @staticmethod
    def _parse_body(body_text: str, toc_entries: list[dict]) -> list[Section]:
        chapters = list(CHAPTER_START.finditer(body_text))

        # pass 1: walk the TOC in order, search forward from a
        # monotonically-advancing cursor for each expected number. same
        # TOC-guided strategy as IPC - see BODY_CANDIDATE_TEMPLATE's
        # comment for why this makes noise (footnotes, brackets, STATE
        # AMENDMENT blocks) harmless rather than something that needs
        # to be stripped out first.
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
            body = body_text[body_start:body_end].strip()

            chapter = CRPCParser._label_for_position(chapters, match.start())
            metadata = {"chapter": chapter, "effective_date": EFFECTIVE_DATE}
            if entry["number"] in CRPC_TO_BNSS:
                metadata["replaced_by"] = CRPC_TO_BNSS[entry["number"]]

            sections.append(Section(
                act=ACT, unit_type="section", number=entry["number"],
                title=entry["title"], body=body, status=DEFAULT_STATUS,
                metadata=metadata,
            ))

        # stub entries (Omitted/Repealed) never had body text to find -
        # record them factually from the TOC's own bracketed text
        for entry in toc_entries:
            if entry["is_stub"]:
                metadata = {"chapter": entry["chapter"], "effective_date": EFFECTIVE_DATE}
                sections.append(Section(
                    act=ACT, unit_type="section", number=entry["number"],
                    title=entry["title"], body=f"[{entry['title']}]",
                    status=DEFAULT_STATUS, metadata=metadata,
                ))

        order = {entry["number"]: i for i, entry in enumerate(toc_entries)}
        sections.sort(key=lambda s: order.get(s.number, len(order)))

        return sections

    @staticmethod
    def _label_for_position(headers: list[re.Match], position: int) -> str:
        current = ""
        for header_match in headers:
            if header_match.start() > position:
                break
            number, title = header_match.group(1), header_match.group(2).strip()
            current = f"Chapter {number}: {title}" if title else f"Chapter {number}"
        return current