from pathlib import Path
import re
import statistics

import pdfplumber
from pdfplumber.page import Page

# footnote reference markers (like the digit after "date" that marks a
# footnote at the bottom of the page) are set in a genuinely smaller font
# in the source PDF - not just a plain digit glued onto a word by
# coincidence. confirmed directly against bns.pdf's character data: the
# "1" in "...on such date1 as the Central Government..." has size 6.96
# against a page body-text size of 11.04, vs. a real "(1)" elsewhere on
# the same page sitting at the full 11.04.
#
# the footnote DEFINITION block at the bottom of the page (the "1. 1st
# day of July, 2024, except..." text that the marker actually points to)
# is ALSO set smaller than body text (roughly 8-9pt against an 11pt body
# in the acts checked so far) - but it isn't superscript-tiny the way the
# in-body marker digit is. if these two "smaller than body" cases aren't
# told apart, every digit inside the footnote text itself (dates,
# notification numbers, section references like "3(ii)") gets
# mis-detected as another marker and bracket-wrapped a second time,
# corrupting the footnote text we're trying to preserve. rather than add
# a second, narrower size ratio (fragile - it'd have to sit exactly
# between "marker" and "footnote block" sizes for every act's PDF), the
# two are told apart structurally: a footnote-definition block always
# starts with a line beginning "N. " and sits in the lower portion of the
# page. once that start line is found, everything below it counts as
# footnote-definition text regardless of its digits' sizes, and only text
# above it is scanned for marker digits.
SUPERSCRIPT_SIZE_RATIO = 0.85

# a footnote-definition entry looks like "1. 1st day of July, 2024,
# except..." - digits, period, space, then real text. requires a real
# following character (not just trailing whitespace) so a lone page
# number doesn't get mistaken for a footnote entry's start.
FOOTNOTE_ENTRY_START = re.compile(r'^\s*(\d{1,3})\.\s+\S')
FOOTNOTE_NUMBER_PREFIX = re.compile(r'^\s*\d{1,3}\.\s*')

# footnote blocks in the acts checked so far sit in the bottom quarter of
# the page, but this is left generous (bottom half) since a page with
# several stacked footnotes could push the block's start line higher -
# false positives are still guarded against by the "smaller than body"
# and "starts with N. " checks below, so being generous here just means
# considering more candidate lines, not accepting bad ones. NOT yet
# verified against every real PDF in the corpus - if a genuine footnote
# block is ever found starting above the page's midpoint, raise this.
FOOTNOTE_REGION_MAX_TOP_FRACTION = 0.5

# how close two adjacent superscript-sized digits need to be (as a
# fraction of character size) to be treated as one multi-digit marker
# number ("12") rather than two separate single-digit markers ("1" then
# "2") that happen to sit next to each other. digits within the same
# marker sit right next to each other with no real gap; a genuine gap
# this small basically never occurs between two unrelated markers, so
# this stays permissive rather than risking splitting a real multi-digit
# marker apart.
MARKER_DIGIT_ADJACENCY_RATIO = 0.5


def _dominant_font_size(pdf: pdfplumber.PDF) -> float:
    """median character size across the whole document - used as the
    "normal body text" baseline every page's digits get compared
    against, so a page that happens to be mostly footnotes (or a mostly-
    blank page) doesn't shift the threshold for everyone else."""
    sizes = [
        char["size"]
        for page in pdf.pages
        for char in page.chars
        if char.get("text", "").strip()
    ]
    return statistics.median(sizes) if sizes else 0.0


def _group_chars_into_lines(chars: list[dict]) -> list[list[dict]]:
    """groups a page's characters into visual lines by rounding each
    char's vertical ("top") position. pdfplumber gives per-character
    positions, not pre-grouped lines, and rounding absorbs sub-pixel
    jitter that would otherwise split one visual line into two groups.
    note: a raised superscript marker character sits at a slightly
    different "top" than the surrounding baseline text it's glued to, so
    it can end up forming its own single-character "line" separate from
    its sentence - that's fine here, since footnote-region detection and
    marker detection both work per-group rather than requiring a marker
    to share a group with the words around it."""
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
    than body text - a numbered clause inside the body (e.g. a real
    "5. Whoever...") fails (c), since only actual footnote text on the
    page is set at the smaller size."""
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


# a bare page-number line ("11" with nothing else on it) - same running
# header/footer noise BNS/BNSS's TOC parsing already has to filter out
# (see bns.py/bnss.py's _clean_title). when it lands right after the
# footnote block's last entry, it's indistinguishable from ordinary
# wrapped continuation text unless explicitly excluded, and would
# otherwise get glued onto that entry's footnote sentence as trailing
# junk (e.g. "...Extraordinary, Part II, sec. 3(ii). 11").
PAGE_NUMBER_LINE = re.compile(r'^\d{1,4}$')


# belt-and-braces on top of PAGE_NUMBER_LINE: that check only catches a
# page number that lands as its own clean line-group. against the real
# PDF, a page number sitting close beneath the footnote block can still
# end up folded into the same reconstructed line as the footnote's last
# line (character-level "top" jitter between two visually distinct lines
# rounds them into the same bucket more often in practice than a
# synthetic test PDF shows), producing something like
# "...sec. 3(ii). 16" as one piece of text - PAGE_NUMBER_LINE never sees
# a line that's cleanly just "16" to reject. this catches that shape
# directly on the assembled footnote text instead of relying on line
# grouping: a bare 1-4 digit number stuck onto the very end, right after
# the sentence's own closing punctuation, is a page number - real
# footnote sentences in these acts consistently end with "." or ")",
# never a bare trailing number, so this is a safe cut rather than a
# guess.
TRAILING_PAGE_NUMBER = re.compile(r'(?<=[.\)])\s+\d{1,4}\s*$')


def _strip_trailing_page_number(text: str) -> str:
    return TRAILING_PAGE_NUMBER.sub("", text)


def _extract_footnote_definitions(lines: list[list[dict]], start_idx: int) -> dict[str, str]:
    """parses the footnote-definition block (lines[start_idx:]) into
    {marker_number: footnote_text}. a definition's text can wrap across
    more than one line, so lines are joined until the next "N. " entry
    or the end of the block. bare page-number lines are dropped rather
    than treated as either a new entry or a continuation of the current
    one, and a trailing page number folded onto the last line is cleaned
    up separately - see TRAILING_PAGE_NUMBER."""
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
    """a digit only counts as a footnote marker if it's BOTH smaller than
    normal body text overall AND smaller than the other, non-digit
    characters on its own line. the first check alone isn't enough: a
    TOC line set entirely in a smaller, uniform font (number and title
    alike) is also "smaller than body baseline" without any digit in it
    actually being a raised superscript marker - confirmed against a
    real IPC TOC, where every section number got wrongly bracketed this
    way, corrupting the TOC past recognition. a genuine marker digit is
    smaller than the NORMAL-sized text sitting right next to it on the
    same line (e.g. "date" at 11pt beside a marker at 7pt); a uniformly-
    small TOC line has no such contrast to find, since its own title
    text is just as reduced as its number."""
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
    """finds runs of adjacent superscript-sized digits in one line (i.e.
    one marker number, possibly multi-digit) and replaces the run with
    the resolved footnote text in {curly braces} - mutating the char
    objects in place, same technique the old bare-digit-bracketing used.
    curly braces rather than square brackets so a resolved footnote can
    never be confused with a real "[...]" the Act's own text uses for
    amended/inserted sections - the two can end up sitting right next to
    each other ("{footnote text}[inserted section text]"). falls back to
    bracing the bare marker number if no footnote definition for it was
    found on this page - never invents footnote text that isn't actually
    there."""
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
    """returns a pdfplumber page where each in-body footnote marker digit
    is replaced by the actual footnote sentence it denotes (looked up
    from that same page's footnote-definition block at the bottom),
    wrapped in {curly braces} - not just the bare bracketed marker
    number, and not square brackets, since those are what the Act's own
    text already uses for amended/inserted sections (the two shapes can
    end up adjacent: "{footnote text}[inserted section text]"). e.g.
    "...on such date1 as..." becomes "...on such date{1st day of July,
    2024, except the provision of sub-section (2) of section 106,...}
    as...". if this page has no footnote block, or a given marker number
    has no matching definition on it, falls back to the bare
    "{number}" bracing - so a missing definition degrades gracefully
    instead of crashing or fabricating text."""
    if baseline <= 0:
        return page

    lines = _group_chars_into_lines(page.chars)
    footnote_start_idx = _find_footnote_region_start(lines, page.height, baseline)

    footnotes: dict[str, str] = {}
    body_line_count = len(lines)
    if footnote_start_idx is not None:
        footnotes = _extract_footnote_definitions(lines, footnote_start_idx)
        body_line_count = footnote_start_idx

        # the footnote block's own text has now been captured into
        # `footnotes` and will be re-inserted inline at each marker -
        # blank it out here so it doesn't ALSO survive as a second, now-
        # redundant copy at the bottom of the page. without this, that
        # trailing text would bleed into whatever section's body happens
        # to span this part of the page (section boundaries are found by
        # searching raw page text, not by any layout awareness), leaving
        # duplicate footnote junk sitting inside stored section bodies.
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
    empty lines behind exactly where it used to sit."""
    return "\n".join(line for line in text.split("\n") if line.strip())


def extract_pdf_text(pdf_path: Path) -> str:
    """
    Extracts text from every page in reading order.
    Resolves in-body footnote markers to the actual footnote sentence
    they denote (see _resolve_footnote_markers), bracketed inline.
    """
    with pdfplumber.open(pdf_path) as pdf:
        baseline = _dominant_font_size(pdf)
        pages = [_drop_blank_lines(_resolve_footnote_markers(page, baseline).extract_text() or "") for page in pdf.pages]
    return "\n".join(pages)


def extract_pdf_pages(pdf_path: Path) -> list[str]:
    """
    Returns one string per page.
    Useful when parsers need page-aware processing
    (e.g. removing TOC or headers). Resolves footnote markers - see
    extract_pdf_text.
    """
    with pdfplumber.open(pdf_path) as pdf:
        baseline = _dominant_font_size(pdf)
        return [_drop_blank_lines(_resolve_footnote_markers(page, baseline).extract_text() or "") for page in pdf.pages]


def remove_repeated_headers(pages: list[str], threshold: float = 0.5) -> list[str]:
    """
    Removes lines repeated on many pages (running headers/footers).
    Parser decides whether to use this.
    """
    if len(pages) < 3:
        return pages

    counts = {}

    for page in pages:
        for line in page.splitlines():
            line = line.strip()
            if line:
                counts[line] = counts.get(line, 0) + 1

    repeated = {
        line
        for line, count in counts.items()
        if count >= len(pages) * threshold
    }

    cleaned = []

    for page in pages:
        kept = [
            line
            for line in page.splitlines()
            if line.strip() not in repeated
        ]
        cleaned.append("\n".join(kept))

    return cleaned


def normalize_whitespace(text: str) -> str:
    """
    Collapses excessive whitespace while preserving paragraph breaks.
    """
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()