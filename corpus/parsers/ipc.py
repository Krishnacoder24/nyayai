"""
IPC parser. fully self-contained - no shared base class or inheritance
with any other act's parser (see issue #26: corpus/parsers/base.py's
unused ChapterSectionParser was deleted, and this act stays independent
of BNS/BNSS/CPC's parsing grammar on purpose). IPC's real PDF has
real-world noise that a generic parser would fight, and none of it is
guaranteed to look anything like BNS/BNSS's (IPC is a 160-year-old,
heavily-amended act; BNS/BNSS are brand new).

"independent" is about not inheriting a shared parsing *class* - it
doesn't mean reinventing plain PDF-reading plumbing. page extraction and
running-header stripping aren't act-specific grammar, so they're shared
via corpus/pdf_utils.py's plain functions instead of copied here.

what the real PDF actually looks like, verified against the real file
(not assumed from IndiaCode's usual formatting):

  - it opens with a ~13-page table of contents (ARRANGEMENT OF SECTIONS)
    listing every section number + title but with NO trailing em-dash -
    if not skipped, these look exactly like section starts.
  - amended/inserted sections are wrapped in "[...]" with a footnote
    marker digit stuck directly in front of the bracket, e.g. "7[5. ...]"
    - the footnote digit is not part of the section number.
  - at least one section number is missing its period entirely
    (seen: "1[17 "Government".—...]" - no "." after "17").
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
from pathlib import Path

from corpus.schemas import Section
from corpus.data.ipc_bns_mapping import IPC_TO_BNS
from corpus.pdf_utils import extract_pdf_pages, remove_repeated_headers

ACT = "IPC"
DEFAULT_STATUS = "repealed"  # BNS replaced the IPC in full, effective 2024-07-01
EFFECTIVE_DATE = "1860-01-01"

# marks where the TOC ends and the actual numbered Act text begins
BODY_START_MARKER = "ACT NO. 45 OF 1860"

# TOC lines: "   9.   Number.\n" or "   10. "Man". "Woman".\n" - no dash, ends at newline
TOC_ENTRY = re.compile(r'\n\s*(\d{1,3}[A-Z]{0,2})\.\s+(.+)')

# chapter headers, in both TOC and body: "CHAPTER XVI" / "CHAPTER VA" (roman + optional letter).
# inserted chapters (VA, IXA, XXA) are, like inserted sections, glued
# directly to a footnote-digit+bracket in the body - "3[CHAPTER VA" - so
# the same atomic optional footnote-digit+bracket unit used below is
# needed here too. without it, this regex simply never matches those
# three chapters at all (not even a partial match on "CHAPTER VA" minus
# the prefix), because "\n\s*CHAPTER" doesn't allow "3[" to sit between
# the newline and the literal text - confirmed against the real PDF,
# where all three letter-suffixed chapters are preceded by exactly this.
CHAPTER_START = re.compile(r'\n\s*(?:\d{1,3}\s*\[)?\s*CHAPTER\s+([IVXLCDM]+[A-Z]?)\s*\n\s*([^\n]*)')

# template for a section-start candidate, parameterised on the EXACT
# number currently expected from the TOC (see _parse_body). searching for
# a specific number - rather than extracting a generic "any digits here"
# candidate list up front and aligning it against the TOC afterwards - is
# what actually makes the TOC-guided approach work in practice.
#
# the generic-candidate-list version was tried first and failed on the
# real PDF: footnote/amendment-history lines at the bottom of a page
# ("6. Illustrations (b), (c) and (d) omitted...", "8. Subs., ibid., for
# section 14.") start with a bare digit + period, exactly like a section
# start. the non-greedy [\s\S]{0,250}? title stretch, finding no dash
# right after the footnote's own text, kept expanding across several more
# footnote lines until it reached the next REAL section's dash further
# down the page - producing one bogus match labeled with the footnote's
# number that swallowed the real section's text inside it. a single one
# of these anywhere in the document silently ate one entire real section
# (verified: this is exactly what happened to section 17, eaten by a
# stray "6." footnote line above it) - and because the candidate-
# consuming loop advanced a single shared pointer and never looked back,
# one such miss permanently exhausted the pointer and every section after
# it in the whole document dropped too (34 of 547 sections survived).
#
# anchoring the search to the literal number currently being looked for
# closes this off entirely: a footnote line reading "6. Illustrations..."
# can never match when we're looking for "17", because "6" != "17" - no
# amount of non-greedy expansion changes what number the regex requires
# up front.
#
# the optional footnote-digit + bracket combo in front is still one
# atomic optional unit (not two independently-optional pieces) for the
# same reason as before: a separately-optional greedy digit prefix would
# happily "match" the leading "1" of a plain "10." as a footnote marker,
# leaving only "0" to satisfy the number being searched for.
#
# the negative lookahead after the number stops "17" from matching inside
# "170" or "17A" - without it, searching for bare "17" could latch onto
# the front of an unrelated longer number instead of the real one.
BODY_CANDIDATE_TEMPLATE = (
    r'(?:^|\n)\s*(?:\d{{1,3}}\s*\[|\[)?\s*{number}(?![A-Za-z0-9])[\s.]{{1,3}}'
    r'(?:[A-Za-z"\u2018\u201c][\s\S]{{0,250}}?)\.\s*[-\u2013\u2014]'
)


def _candidate_pattern(number: str) -> re.Pattern:
    return re.compile(BODY_CANDIDATE_TEMPLATE.format(number=re.escape(number)), re.MULTILINE)


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
        # shared with every other act's parser via corpus/pdf_utils.py -
        # IPC used to keep its own private copy of this exact logic, which
        # is the duplicate implementation issue #26 flagged. there's
        # nothing IPC-specific about "strip lines repeated across most
        # pages" - it's page/PDF plumbing, not act-specific grammar, so it
        # belongs in the shared plain-function module, not copied into
        # each parser. IPC still doesn't inherit from any shared parser
        # *class* - that's the actual independence this file's docstring
        # cares about - it just isn't reinventing plain PDF-reading helpers.
        pages = extract_pdf_pages(pdf_path)
        pages = remove_repeated_headers(pages)
        return "\n".join(pages)

    @staticmethod
    def _split_toc_and_body(raw_text: str) -> tuple[str, str]:
        marker_pos = raw_text.find(BODY_START_MARKER)
        if marker_pos == -1:
            raise ValueError(
                f"couldn't find '{BODY_START_MARKER}' - IPC PDF layout may have changed, "
                f"check where the table of contents ends"
            )
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