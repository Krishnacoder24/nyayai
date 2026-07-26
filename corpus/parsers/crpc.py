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

# marks where the real numbered-section body ends and the Schedules
# begin - CrPC has no equivalent in ipc.py, since IPC has no Schedules
# of its own. confirmed via direct execution against crpc.pdf that this
# phrase occurs exactly once in the whole document, right where "THE
# FIRST SCHEDULE / CLASSIFICATION OF OFFENCES" (the offence-type table)
# starts. without cutting body_text off here, section 484 ("Repeal and
# savings" - the real last section) has no way to know where its own
# body should end, since nothing in the existing TOC-guided search logic
# tells it to stop before non-section content - its body silently
# absorbed the entire First Schedule table AND the Second Schedule
# (~240 pages) as if all of it were part of section 484's own text. that
# table also has its own row-numbers (172, 173, 334, 335, ...) typeset
# in a smaller font than body text, which get misidentified as
# superscript footnote markers by _is_marker_digit - trimming the
# Schedules out of body_text entirely avoids that misidentification too,
# rather than needing a separate fix for it.
SCHEDULE_START_PATTERN = re.compile(r'\n\s*THE\s+FIRST\s+SCHEDULE\b', re.IGNORECASE)

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

# separate, more lenient ratio specifically for footnote-BLOCK text (as
# opposed to SUPERSCRIPT_SIZE_RATIO above, which is for the marker
# DIGIT - a genuine superscript, much smaller). confirmed via direct
# execution against real crpc.pdf data that these two need different
# thresholds: the real footnote-definition block sits at 9.0pt against
# a 9.96pt body baseline (a 0.90 ratio) - ABOVE SUPERSCRIPT_SIZE_RATIO's
# 0.85 cutoff, which caused a real footnote block to be rejected
# outright (see _find_footnote_region_start's docstring). this also
# needs to be a REQUIRED check, not something that can be skipped just
# because a separator rule was found: confirmed a real section's own
# body-start line ("321. Withdrawal from prosecution.—...", at full
# 9.96pt body size, ratio 1.0) sitting below an unrelated rule
# elsewhere on the same page (that page's own separate footnote
# separator) got misread as a footnote-block start when the rule check
# was trusted alone, silently blanking out that section's entire real
# body. 0.95 sits cleanly between the two confirmed real ratios (0.90
# for genuine footnote-block text, 1.0 for genuine body text).
FOOTNOTE_BLOCK_SIZE_RATIO = 0.95
FOOTNOTE_ENTRY_START = re.compile(r'^\s*(\d{1,3})\.\s*\S')
FOOTNOTE_NUMBER_PREFIX = re.compile(r'^\s*\d{1,3}\.\s*')
FOOTNOTE_REGION_MAX_TOP_FRACTION = 0.5

# a genuine footnote-separator rule confirmed present in the real CrPC
# PDF (screenshot: a plain horizontal line sits directly above every
# footnote block, page number below it) - not a full page-width rule,
# roughly a quarter to a third of it in the sample seen. kept generous
# (0.15) since the exact width wasn't measured against real coordinates,
# only eyeballed from a screenshot - a short table-border artifact could
# in principle still clear this bar, but MIN_RULE_WIDTH_FRACTION exists
# to reject trivial/degenerate shapes, not to precisely fingerprint
# "the" separator; the position check (bottom half of the page) and the
# "N. " text-shape check on the line below it still have to hold too.
MIN_RULE_WIDTH_FRACTION = 0.15
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


def _find_footnote_rule_top(page: Page, page_height: float) -> float | None:
    """returns the vertical position of the horizontal separator rule
    that sits directly above the footnote-definition block, or None if
    this page has no such rule. confirmed present in the real CrPC PDF -
    a plain drawn horizontal line sits directly above every footnote
    block (the page number sits below the block, not near this rule).

    checks both page.lines and page.rects, since a "line" in a PDF is
    sometimes a genuine hairline object and sometimes a very thin filled
    rectangle depending on how the document was generated - pdfplumber
    exposes them as two different object types and there's no reliable
    way to know in advance which one a given PDF used.

    a candidate only counts if it's near-horizontal (top and bottom
    edge within a hair of each other), wide enough to plausibly be a
    section-separator rather than a stray table-border fragment (see
    MIN_RULE_WIDTH_FRACTION), and sits in the bottom half of the page -
    the same region the footnote block itself is expected in. if more
    than one such rule is found, the topmost one is used, since the
    footnote block starts right after the first rule it crosses, not a
    later one further down.
    """
    candidates: list[float] = []

    for shape in list(page.lines) + list(page.rects):
        top = shape.get("top")
        if top is None or top < page_height * FOOTNOTE_REGION_MAX_TOP_FRACTION:
            continue
        height = shape.get("height", 0)
        width = shape.get("width", 0)
        if height > 2:  # not near-horizontal - too tall to be a rule
            continue
        if width < page.width * MIN_RULE_WIDTH_FRACTION:
            continue
        candidates.append(top)

    return min(candidates) if candidates else None


def _find_footnote_region_start(
    lines: list[list[dict]], page_height: float, baseline: float, rule_top: float | None = None
) -> int | None:
    """returns the index into `lines` where the footnote-definition block
    starts, or None if this page has no footnote block. a line only
    counts as the start of one if it (a) begins with the "N. " shape,
    (b) sits in the bottom portion of the page, and (c) is set smaller
    than body text (see FOOTNOTE_BLOCK_SIZE_RATIO) - all three always
    required, regardless of whether a separator rule was found.

    when a rule WAS found (see _find_footnote_rule_top), the candidate
    must ALSO sit at or below it - an extra, not a replacement, for the
    three checks above.

    earlier versions of this function dropped the size check whenever a
    rule was found, on the theory that a confirmed drawn rule was
    reliable enough on its own. confirmed via direct execution against
    crpc.pdf that this was wrong: a real section's own body-start line
    ("321. Withdrawal from prosecution.—...", at full 9.96pt body size)
    happened to sit below an unrelated rule elsewhere on the same page
    (that page's own separate, genuine footnote separator, for
    different content earlier on the page) and got misread as a
    footnote-block start - silently blanking out that entire section's
    real body. the size check is what would have caught it (its ratio
    is 1.0, body-sized, not the ~0.90 a genuine footnote block sits at)
    and there's no way to safely skip it just because a rule exists
    somewhere on the page - a page can have more than one drawn rule for
    reasons unrelated to marking A footnote block.
    """
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
        if avg_size >= baseline * FOOTNOTE_BLOCK_SIZE_RATIO:
            continue
        if rule_top is not None and top < rule_top:
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


def _extract_page_footnotes(page: Page, baseline: float) -> tuple[list[list[dict]], int, dict[str, str]]:
    """returns (lines, body_line_count, footnotes) for one page - the
    footnote-DEFINITION half of what used to be _resolve_footnote_markers,
    split out so definitions can be looked up across a page boundary
    (see _resolve_footnote_markers_for_document) before any marker gets
    resolved. blanks out the footnote-definition block's own characters
    in place, same as before - body_line_count marks where the real body
    ends and the (now blanked) footnote block begins."""
    lines = _group_chars_into_lines(page.chars)
    if baseline <= 0:
        return lines, len(lines), {}

    rule_top = _find_footnote_rule_top(page, page.height)
    footnote_start_idx = _find_footnote_region_start(lines, page.height, baseline, rule_top)

    footnotes: dict[str, str] = {}
    body_line_count = len(lines)
    if footnote_start_idx is not None:
        footnotes = _extract_footnote_definitions(lines, footnote_start_idx)
        body_line_count = footnote_start_idx

        for line in lines[footnote_start_idx:]:
            for ch in line:
                ch["text"] = ""

    # a bare running page-number line (just digits, nothing else) can
    # sit at the bottom of ANY page, whether or not that page happens to
    # have a footnote block at all - confirmed via direct execution: a
    # standalone "195" leaked straight into section 484's body on a page
    # with no footnote block, since the earlier version of this function
    # only ever stripped a page number that happened to fall inside an
    # already-detected footnote region. blank out any such line in the
    # bottom portion of the page independently of footnote detection.
    for line in lines[:body_line_count]:
        if not line:
            continue
        top = min(c["top"] for c in line)
        if top < page.height * FOOTNOTE_REGION_MAX_TOP_FRACTION:
            continue
        if PAGE_NUMBER_LINE.match(_line_text(line).strip()):
            for ch in line:
                ch["text"] = ""

    return lines, body_line_count, footnotes


def _resolve_footnote_markers_for_document(pdf: pdfplumber.PDF, baseline: float) -> list[Page]:
    """resolves footnote markers across the WHOLE document rather than
    strictly per-page.

    confirmed against real CrPC output that a marker and its own
    footnote definition can end up on two different physical PDF pages -
    a marker sitting near the very end of one page ("(2) It extends to
    the whole of India {1}***:", immediately followed by a multi-clause
    proviso that fills out the rest of that page), with its definition
    pushed onto the bottom of the NEXT page because there wasn't room to
    fit it under the first page's own content. resolving strictly
    per-page (the original approach here, matching ipc.py) misses this
    two ways at once: the marker falls back to the bare "{number}"
    placeholder since its own page's footnotes dict is empty, AND the
    real definition text never gets blanked out - it's sitting past
    whatever body_line_count boundary its OWN page computed, so it shows
    up as leftover body prose glued onto whatever section happens to
    start right after it on that later page.

    strategy: extract every page's own (lines, body_line_count,
    footnotes) first, in one pass, via _extract_page_footnotes - this
    also means every genuine footnote block still gets blanked out of
    its own page regardless of which page's markers end up using it.
    then resolve each page's markers using that page's OWN footnotes
    dict first; only if a marker number isn't found there, fall back to
    the IMMEDIATELY NEXT page's footnotes dict. deliberately a bounded,
    single-page lookahead rather than merging every page's footnotes
    into one whole-document dict - footnote numbering very likely
    restarts per page (both this page's "1" and a much later page's
    "1" are plausible), so a document-wide merge would risk substituting
    a completely unrelated page's same-numbered footnote into a marker
    it has nothing to do with. one page ahead is as far as the evidence
    seen so far justifies reaching.
    """
    pages_data = [
        {"page": page, **dict(zip(("lines", "body_line_count", "footnotes"), _extract_page_footnotes(page, baseline)))}
        for page in pdf.pages
    ]

    for i, data in enumerate(pages_data):
        combined_footnotes = dict(data["footnotes"])
        if i + 1 < len(pages_data):
            # this page's own definitions always take priority; only
            # fill in numbers this page doesn't already have
            for number, text in pages_data[i + 1]["footnotes"].items():
                combined_footnotes.setdefault(number, text)

        for line in data["lines"][: data["body_line_count"]]:
            _resolve_markers_in_line(line, baseline, combined_footnotes)

    return [d["page"] for d in pages_data]


def _drop_blank_lines(text: str) -> str:
    """removes lines that are empty or whitespace-only. blanking a
    footnote-definition line's characters (see _extract_page_footnotes)
    only empties the characters, not the line's vertical space -
    pdfplumber's extract_text() still emits a blank line there, since it
    places line breaks by character position, not by whether any text
    survived. without this, every footnote block leaves a visible gap of
    empty lines behind exactly where it used to sit. same helper as
    pdf_utils.py's version - duplicated rather than imported since it's
    private there and this file follows the same self-contained
    convention as ipc.py."""
    return "\n".join(line for line in text.split("\n") if line.strip())



# marks the start of a state-specific amendment block, interleaved
# directly in the body between one section's real end and the next
# section's start - NOT a footnote, NOT bracketed, just plain uppercase
# body text (e.g. "STATE AMENDMENT\nHaryana\nIn the Code of..."). CAUTION
# was flagged and left unhandled in an earlier draft of this file (see
# git history) - confirmed via direct execution against the real PDF
# that a section's real content ALWAYS ends cleanly before the first
# "STATE AMENDMENT"/"STATE AMENDMENTS" occurrence within its own matched
# range (both header spellings appear in the wild - confirmed 6
# sections use the plural "STATE AMENDMENTS", e.g. 125, 127, 167), and
# real central-act text never resumes after it within that same section
# (checked section 24, which has TWO separate "STATE AMENDMENT" blocks
# back to back - Karnataka/Maharashtra/Madhya Pradesh/West Bengal x2,
# then a second header for Jammu and Kashmir - with no real Act text
# between or after them). safe to truncate a section's body at the
# FIRST occurrence of this marker within its own range, same idea as
# BODY_END_MARKER-style truncation elsewhere in the other act parsers.
STATE_AMENDMENT_BLOCK = re.compile(r'\n\s*STATE AMENDMENTS?\s*\n')


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
            resolved_pages = _resolve_footnote_markers_for_document(pdf, baseline)
            pages = [_drop_blank_lines(page.extract_text() or "") for page in resolved_pages]
        pages = remove_repeated_headers(pages)
        return "\n".join(pages)

    @staticmethod
    def _split_toc_and_body(raw_text: str) -> tuple[str, str]:
        toc_text, body_text = CRPCParser._find_toc_body_split(raw_text)
        return toc_text, CRPCParser._trim_schedules(body_text)

    @staticmethod
    def _trim_schedules(body_text: str) -> str:
        """cuts body_text off at the start of the Schedules, if found -
        see SCHEDULE_START_PATTERN's comment for why this matters (the
        real last section otherwise has no bound on where its own body
        ends). if the pattern isn't found (a different edition phrases
        it differently, or this Act version genuinely has no Schedules
        included), returns body_text unchanged rather than guessing at a
        cutoff that isn't actually there."""
        match = SCHEDULE_START_PATTERN.search(body_text)
        return body_text[: match.start()] if match else body_text

    @staticmethod
    def _find_toc_body_split(raw_text: str) -> tuple[str, str]:
        enacting_match = ENACTING_CLAUSE_PATTERN.search(raw_text)
        if enacting_match is not None:
            # unlike IPC (where the real "CHAPTER I" heading appears
            # BEFORE the preamble text it introduces, so backing up from
            # the preamble match to find it makes sense), CrPC's real
            # Chapter I heading appears AFTER its enacting clause:
            # "...as follows:-\nCHAPTER I\nPRELIMINARY\n1. Short title...".
            #
            # confirmed via direct execution against the real crpc.pdf
            # that copying IPC's backward rfind() here was a real bug,
            # not just a theoretical risk: searching BACKWARD from the
            # enacting-clause match found the TOC's OWN last chapter
            # heading ("CHAPTER XXXVII\nMISCELLANEOUS") instead of the
            # real body's first one - the nearest "\nCHAPTER" text
            # before the enacting clause is still inside the TOC, not
            # past it. that silently split the document mid-TOC: section
            # entries 474 through 484, plus both Schedules, all ended up
            # inside body_text instead of toc_text, which meant they
            # were never in toc_entries to search for at all - and
            # everything from that point to the end of the document
            # (roughly 240 pages, including the entire First Schedule
            # offence-classification table) got silently absorbed into
            # whichever section happened to be matched last (473),
            # since nothing told _parse_body to stop looking there.
            chapter_pos = raw_text.find("\nCHAPTER", enacting_match.end())
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

            # cut off state-specific amendment text before it ever
            # becomes part of the stored body - see STATE_AMENDMENT_BLOCK's
            # comment for why the FIRST occurrence within this range is
            # always the right cutoff point.
            state_amendment_match = STATE_AMENDMENT_BLOCK.search(body_text, body_start, body_end)
            if state_amendment_match:
                body_end = state_amendment_match.start()

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