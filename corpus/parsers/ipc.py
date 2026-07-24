"""
IPC parser. fully self-contained - no shared base class or inheritance
with any other act's parser (see issue #26: corpus/parsers/base.py's
unused ChapterSectionParser was deleted, and this act stays independent
of BNS/BNSS/CPC's parsing grammar on purpose). IPC's real PDF has
real-world noise that a generic parser would fight, and none of it is
guaranteed to look anything like BNS/BNSS's (IPC is a 160-year-old,
heavily-amended act; BNS/BNSS are brand new).

"independent" is about not inheriting a shared parsing *class* - it
doesn't mean reinventing plain PDF-reading plumbing. running-header
stripping is shared via corpus/pdf_utils.py, but IPC-specific footnote
resolution lives here.

what the real PDF actually looks like, verified against the real file
(not assumed from IndiaCode's usual formatting):

  - it opens with a ~13-page table of contents (ARRANGEMENT OF SECTIONS)
    listing every section number + title but with NO trailing em-dash -
    if not skipped, these look exactly like section starts.
  - amended/inserted sections are wrapped in "[...]" with a footnote
    marker digit stuck directly in front of the bracket. this parser
    resolves each in-body footnote marker to the actual footnote sentence
    it denotes (pulled from that page's own footnote-definition block),
    wrapped in {curly braces} to differentiate from normal [] brackets in
    the PDF. so the shape becomes: "{Subs. by Act 26 of 1955, s. 117 and
    Sch., for section 5.}[5. ...]" - the resolved footnote in {}, then the
    real section-opening bracket.
  - at least one section number is missing its period entirely
    (seen: "1{17} "Government".—...]" - no "." after "17").
  - repealed/omitted sections (13, 15, 16, 56, 58, 59, 61, 62, ...) have
    NO body text at all - the body just skips straight from section 12
    to section 17.
  - chapter headers can have a letter suffix ("CHAPTER VA", "CHAPTER IXA").

approach: use the TOC as ground truth. it lists every section number in
order, so instead of trusting a regex to correctly identify every section
start in noisy text, we walk the body looking only for the number
currently expected next (per the TOC's sequence). this makes footnote/
bracket noise harmless - a false-positive match is simply never the
number being waited for, so it's silently skipped rather than accepted.
"""

import re
import statistics
from pathlib import Path

import pdfplumber
from pdfplumber.page import Page

from corpus.schemas import Section
from corpus.data.ipc_bns_mapping import IPC_TO_BNS
from corpus.pdf_utils import remove_repeated_headers

ACT = "IPC"
DEFAULT_STATUS = "repealed"  # BNS replaced the IPC in full, effective 2024-07-01
EFFECTIVE_DATE = "1860-01-01"

# marks where the TOC ends and the actual numbered Act text begins.
#
# empirically confirmed against the real PDF's own extracted text (ran a
# direct diagnostic against it, not assumed): "ACT NO. 45 OF 1860"
# appears in this document only ONCE - not duplicated on a separate cover
# page the way BNS/BNSS's copy of their own act-number line is - and that
# one occurrence sits only ~1,400 characters before the real body, glued
# directly to a very long resolved footnote (the territorial-extension
# history): "...ACT NO. 45 OF 1860[<long footnote>]\nCHAPTER I\n
# INTRODUCTION\nPreamble.\u2014WHEREAS it is expedient to provide a
# general Penal Code...". using the act-number line as the split point
# works, but leaves that entire footnote sitting awkwardly right at the
# front of body_text.
#
# the preamble text itself is used as the primary marker instead - it's
# textually adjacent to the real Chapter I / Section 1 start, confirmed
# to occur exactly once, and doesn't depend on the "ACT NO...." line
# existing at all (worth keeping the old marker as a fallback in case a
# different edition phrases its preamble differently or omits it).
PREAMBLE_MARKER_PATTERN = re.compile(
    r'WHEREAS\s+it\s+is\s+expedient\s+to\s+provide\s+a\s+general\s+Penal\s+Code',
    re.IGNORECASE,
)
BODY_START_MARKER_PATTERN = re.compile(r'ACT\s*NO\.?\s*45\s*OF\s*1860', re.IGNORECASE)

# TOC lines: "   9.   Number.\n" or "   10. "Man". "Woman".\n" - no dash, ends at newline
TOC_ENTRY = re.compile(r'\n\s*(\d{1,3}[A-Z]{0,2})\.\s+(.+)')

# chapter headers, in both TOC and body: "CHAPTER XVI" / "CHAPTER VA" (roman + optional letter).
# inserted chapters (VA, IXA, XXA) are, like inserted sections, glued
# directly to a footnote marker in the body. footnotes are now in {text} format.
# updated to match: a braced chunk of arbitrary footnote text followed by
# the real opening bracket (\{[^\}\n]*\}\s*\[).
CHAPTER_START = re.compile(r'\n\s*(?:\{[^\}\n]*\}\s*\[)?\s*CHAPTER\s+([IVXLCDM]+[A-Z]?)\s*\n\s*([^\n]*)')

# template for a section-start candidate, parameterised on the EXACT
# number currently expected from the TOC (see _parse_body). searching for
# a specific number - rather than extracting a generic "any digits here"
# candidate list up front and aligning it against the TOC afterwards - is
# what actually makes the TOC-guided approach work in practice.
#
# the optional footnote + real-bracket combo in front matches the format:
# {footnote text}[section number. Title.—Body]
# The footnote (in curly braces) appears before the opening bracket.
#
# the negative lookahead after the number stops "17" from matching inside
# "170" or "17A" - without it, searching for bare "17" could latch onto
# the front of an unrelated longer number instead of the real one.
BODY_CANDIDATE_TEMPLATE = (
    r'(?:^|\n)\s*(?:\{{[^\}}\n]*\}}\s*)?(\[)?\s*{number}(?![A-Za-z0-9])[\s.]{{1,3}}'
    r'(?:[A-Za-z"\u2018\u201c][\s\S]{{0,250}}?)\.\s*[-\u2013\u2014]'
)


def _candidate_pattern(number: str) -> re.Pattern:
    return re.compile(BODY_CANDIDATE_TEMPLATE.format(number=re.escape(number)), re.MULTILINE)


SUPERSCRIPT_SIZE_RATIO = 0.85
FOOTNOTE_ENTRY_START = re.compile(r'^\s*(\d{1,3})\.\s+\S')
FOOTNOTE_NUMBER_PREFIX = re.compile(r'^\s*\d{1,3}\.\s*')
FOOTNOTE_REGION_MAX_TOP_FRACTION = 0.5
MARKER_DIGIT_ADJACENCY_RATIO = 0.5
FOOTNOTE_SEPARATOR_GAP_RATIO = 2.0
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


def _typical_line_gap(lines: list[list[dict]]) -> float:
    tops = [min(c["top"] for c in line) for line in lines if line and _line_text(line).strip()]
    gaps = [b - a for a, b in zip(tops, tops[1:]) if b > a]
    return statistics.median(gaps) if gaps else 0.0


def _find_footnote_region_start(lines: list[list[dict]], page_height: float, baseline: float) -> int | None:
    if baseline <= 0:
        return None

    typical_gap = _typical_line_gap(lines)

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

        prev_idx = i - 1
        while prev_idx >= 0 and not _line_text(lines[prev_idx]).strip():
            prev_idx -= 1
        if prev_idx < 0:
            continue
        prev_top = min(c["top"] for c in lines[prev_idx])
        gap = top - prev_top
        if typical_gap <= 0 or gap < typical_gap * FOOTNOTE_SEPARATOR_GAP_RATIO:
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


# TOC titles that mean "this section has no body text at all"
STUB_MARKERS = ("[omitted", "[repealed")


class IPCParser:
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
            pages = [_resolve_footnote_markers(page, baseline).extract_text() or "" for page in pdf.pages]
        pages = remove_repeated_headers(pages)
        return "\n".join(pages)

    @staticmethod
    def _split_toc_and_body(raw_text: str) -> tuple[str, str]:
        preamble_match = PREAMBLE_MARKER_PATTERN.search(raw_text)
        if preamble_match is not None:
            # back up to the nearest preceding "CHAPTER" heading so it's
            # kept in body_text (where CHAPTER_START can find it) instead
            # of being stranded in toc_text - the preamble text is what's
            # confirmed unique, the real Chapter I heading immediately
            # precedes it.
            chapter_pos = raw_text.rfind("\nCHAPTER", 0, preamble_match.start())
            marker_pos = chapter_pos if chapter_pos != -1 else preamble_match.start()
            return raw_text[:marker_pos], raw_text[marker_pos:]

        # fallback: this edition doesn't phrase its preamble the expected
        # way - try the act-number line instead. last match, not first,
        # in case IT is duplicated on a separate cover page in whatever
        # edition ended up here.
        matches = list(BODY_START_MARKER_PATTERN.finditer(raw_text))
        if not matches:
            raise ValueError(
                "couldn't find IPC's body-start point - neither the preamble "
                "('WHEREAS it is expedient...') nor an 'ACT NO. 45 OF 1860'-style "
                "marker were found anywhere in the extracted text. IPC PDF layout "
                "may have changed - check the text right after the table of "
                "contents ends and update PREAMBLE_MARKER_PATTERN / "
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
            chapter = IPCParser._label_for_position(chapters, match.start())
            is_stub = title.lower().startswith(STUB_MARKERS)
            entries.append({"number": number, "title": title, "chapter": chapter, "is_stub": is_stub})

        return entries

    @staticmethod
    def _parse_body(body_text: str, toc_entries: list[dict]) -> list[Section]:
        chapters = list(CHAPTER_START.finditer(body_text))

        # pass 1: walk the TOC in order, and for each expected (non-stub)
        # number, search FORWARD from a monotonically-advancing cursor for
        # a candidate matching that exact number. this is the actual
        # TOC-guided approach the generic candidate-list version was
        # supposed to implement: search for what's expected next, not
        # "grab whatever's next and hope it matches". footnote/bracket
        # noise is harmless by construction, because the regex itself
        # requires the exact number - it can never accidentally match on
        # an unrelated digit the way a generic "any number" scan can.
        #
        # if a number genuinely isn't found (a real gap not caught by
        # STUB_MARKERS), the cursor simply doesn't advance and the next
        # entry searches from the same place - one miss doesn't cascade
        # into every later section being dropped, unlike the old version.
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

        # pass 2: build Sections for matched entries, using the next
        # matched entry's start as this entry's body end
        matched_positions = sorted(matched.items())
        sections = []

        for pos, (i, match) in enumerate(matched_positions):
            entry = toc_entries[i]
            body_start = match.end()
            body_end = matched_positions[pos + 1][1].start() if pos + 1 < len(matched_positions) else len(body_text)
            body = body_text[body_start:body_end].strip()

            chapter = IPCParser._label_for_position(chapters, match.start())
            metadata = {"chapter": chapter, "effective_date": EFFECTIVE_DATE}
            if entry["number"] in IPC_TO_BNS:
                metadata["replaced_by"] = IPC_TO_BNS[entry["number"]]

            sections.append(Section(
                act=ACT, unit_type="section", number=entry["number"],
                title=entry["title"], body=body, status=DEFAULT_STATUS,
                metadata=metadata,
            ))

        # stub entries (Omitted/Repealed) never had body text to find -
        # record them factually from the TOC's own bracketed text, nothing invented
        for entry in toc_entries:
            if entry["is_stub"]:
                metadata = {"chapter": entry["chapter"], "effective_date": EFFECTIVE_DATE}
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
        """finds the chapter header that comes right before this position in the text."""
        current = ""
        for header_match in headers:
            if header_match.start() > position:
                break
            number, title = header_match.group(1), header_match.group(2).strip()
            current = f"Chapter {number}: {title}" if title else f"Chapter {number}"
        return current