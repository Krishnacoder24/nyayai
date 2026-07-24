"""
BNS parser. fully self-contained - no shared base class or inheritance
with any other act's parser (see issue #26). BNS is a brand-new act
(2023, in force from 2024) with almost none of IPC's 160 years of
amendment scar tissue, so its real PDF is a lot cleaner than IPC's - but
"cleaner" doesn't mean "identical", and a couple of its own quirks still
needed empirical fixing (see below). this file doesn't reuse IPC's
regexes as-is for that reason - same *approach* (TOC as ground truth),
different patterns tuned against BNS's own PDF.

page extraction and running-header stripping are shared plumbing, not
act-specific grammar, so they still come from corpus/pdf_utils.py rather
than being reimplemented here - same reasoning as IPC's parser.

what the real PDF actually looks like, verified against the real file
(corpus/sources/bns/, "[As on the 6th October, 2025]" edition):

  - it opens with a cover page, an abbreviations page, then a ~13-page
    table of contents (ARRANGEMENT OF SECTIONS) - same shape as IPC's,
    section number + title with NO trailing dash, which is what makes
    TOC entries hard to tell apart from real section starts by regex
    alone (hence the TOC-guided approach, same as IPC).
  - section numbers run 1-358 with NO gaps and NO letter suffixes at all
    - verified by diffing the full number sequence against range(1, 359).
    makes sense: this is the first edition of a new act, there's been no
    amendment cycle yet to insert/repeal anything.
  - almost no amendment-history noise: the entire document has exactly
    ONE footnote in it (marking the 1-July-2024 commencement date on
    section 1) and exactly ONE bracket pair in the whole body (the
    enactment date "[25th December, 2023.]"). IPC's bracket/footnote
    noise simply isn't a live problem for BNS yet - confirmed by
    counting occurrences directly rather than assuming. NOTE: that one
    commencement footnote is exactly the character data
    corpus/pdf_utils.py's _format_superscripts() was verified against
    (it's what "confirmed directly against bns.pdf's character data" in
    that file's docstring refers to) - the superscript "1" marking the
    date now comes through as its own "[1]" rather than a bare digit
    glued onto the word, and the optional footnote-prefix patterns below
    are written to match that bracketed shape, not the old raw one.
  - BUT: two sections break the otherwise-uniform "number. Title.—Body"
    format, found by running IPC's original template against the real
    text and checking what didn't match:
      * section 217 has no period before its dash ("...to injury of
        another person—Whoever gives...") - title runs straight into a
        single dash with nothing separating them.
      * section 255 has its dash immediately after the number
        ("255.—Public servant disobeying...") rather than after the
        title - the title itself is followed by a second, ordinary
        dash before the body. so the shape is "N.—Title.—Body" instead
        of "N. Title.—Body" for this one section.
    both are handled by making the pre-title dash optional and the
    post-title period optional in the candidate pattern, rather than
    hand-carving a special case for two section numbers.
  - the TOC has its own page-break noise: a bare page-number line (e.g.
    "\n11\n") sometimes lands in the middle of a wrapped title, and the
    running "SECTIONS" header repeats at the top of most TOC pages. the
    original single-line IPC-style TOC regex doesn't reproduce this,
    because unlike IPC's title text, some BNS titles wrap across a page
    boundary rather than just a line break, and the page-break debris
    sits between the wrapped halves.
  - after section 358 ("Repeal and savings") the document keeps going
    into a "STATEMENT OF OBJECTS AND REASONS" section, which is not Act
    text - has to be excluded or section 358 silently swallows it as
    its own body.

approach: same TOC-guided idea as IPC - walk the TOC in order, and for
each expected number, search FORWARD from a monotonically-advancing
cursor for a candidate matching that exact number. see ipc.py for the
full rationale on why this beats a generic "find every candidate, then
align" pass. verified end to end against the real PDF: all 358 of 358
TOC entries find a body match, in order, with sane boundaries.
"""

import re
from pathlib import Path

from corpus.schemas import Section
from corpus.pdf_utils import extract_pdf_pages, remove_repeated_headers

ACT = "BNS"
DEFAULT_STATUS = "active"  # BNS is the currently-in-force act (replaced IPC 2024-07-01)

# confirmed via the Act's own footnote on section 1 ("1. 1st day of July,
# 2024, ... vide notification No. S.O. 850(E)...") - not assumed from
# memory of the news coverage, read directly out of the PDF, since a
# wrong effective date here is exactly the kind of thing that could
# mislead a court or law firm.
EFFECTIVE_DATE = "2024-07-01"

# marks where the TOC ends and the actual numbered Act text begins.
# "ACT NO. 45 OF 2023" also appears once on the cover page, so that
# string alone isn't unique enough to split on (see _split_toc_and_body) -
# this exact phrase, on the other hand, only occurs once in the whole
# document, right where Chapter I of the real body starts.
BODY_START_MARKER = "BE it enacted by Parliament"

# after section 358 the PDF continues into non-Act commentary. without
# cutting the body off here, section 358 (the last TOC entry, with no
# next entry to bound it) would swallow this whole section as its own body.
BODY_END_MARKER = "STATEMENT OF OBJECTS AND REASONS"

# TOC entries: "9. Limit of punishment...\n" (no dash, like IPC) but titles
# can wrap across a page break, not just a line break, so the stop
# condition needs to cover more than "next line starts with a number":
#   - the next real entry ("\d+.")
#   - a chapter header
#   - a "Of ..." sub-heading followed directly by a number (mirrors IPC)
#   - a bare page-number line ("\n11\n") - this is what actually differs
#     from IPC and is what section 255's title was bleeding into before
#     this alternative was added (confirmed by re-running against the
#     real TOC text with and without it)
#   - end of the TOC text, for the very last entry
TOC_ENTRY = re.compile(
    r'\n(\d{1,3})\.\s+([\s\S]+?)'
    r'(?=\n\d{1,3}\.\s|\nCHAPTER\s|\n[A-Z][a-z][^\n]*\n\d|\n\d{1,4}\n|\Z)'
)

# chapter headers, in both TOC and body: "CHAPTER XX\nREPEAL AND SAVINGS".
# no letter-suffixed chapters exist in this edition (nothing's been
# inserted yet), but the optional footnote-bracket prefix is kept anyway
# since it costs nothing and IPC needed the exact same shape once its
# first amendment landed - cheap insurance against the next edition.
# shape is "[N][" (bracketed footnote number, then the real opening
# bracket), matching pdf_utils._format_superscripts()'s output, not the
# old bare "N[" shape - see this file's top docstring.
CHAPTER_START = re.compile(r'\n\s*(?:\[\d{1,3}\]\s*\[)?\s*CHAPTER\s+([IVXLCDM]+[A-Z]?)\s*\n\s*([^\n]*)')

# section-start candidate, parameterised on the exact number currently
# expected from the TOC - same TOC-guided reasoning as IPC (see ipc.py
# for why searching for a specific number beats a generic candidate scan).
#
# two pieces are optional here that IPC's template didn't need:
#   - an optional dash right after the number, to cover section 255's
#     "255.—Title.—Body" shape (dash before the title, not just after it)
#   - an optional period before the final dash, to cover section 217's
#     "...another person—Whoever..." (no period at all before the dash)
# both were added only after confirming, by running the stricter
# IPC-style template first, that these two sections were the entire
# gap (356/358 matched without them, 358/358 with them) - not guessed
# up front.
#
# the leading optional footnote-prefix is "\[\d{1,3}\]\s*\[" (bracketed
# footnote number, then the real opening bracket), matching what
# pdf_utils._format_superscripts() actually emits - not the old bare
# "\d{1,3}\s*\[" shape. this is the one section (1, via its commencement
# footnote) where that prefix is exercised for real in this edition.
BODY_CANDIDATE_TEMPLATE = (
    r'(?:^|\n)\s*(?:\[\d{{1,3}}\]\s*\[|\[)?\s*{number}(?![A-Za-z0-9])[\s.]{{1,3}}[-\u2013\u2014]?\s*'
    r'(?:[A-Za-z"\u2018\u201c][\s\S]{{0,250}}?)\.?\s*[-\u2013\u2014]'
)


def _candidate_pattern(number: str) -> re.Pattern:
    return re.compile(BODY_CANDIDATE_TEMPLATE.format(number=re.escape(number)), re.MULTILINE)


# TOC titles that mean "this section has no body text at all" - none
# exist in this edition (no gaps in 1-358), but kept for parity with IPC
# and because a future amended edition of BNS could very plausibly
# introduce a repealed/omitted section the same way IPC's did.
STUB_MARKERS = ("[omitted", "[repealed")


class BNSParser:
    act = ACT

    def parse(self, pdf_path: Path) -> list[Section]:
        raw_text = self._extract_raw_text(pdf_path)
        toc_text, body_text = self._split_toc_and_body(raw_text)
        toc_entries = self._parse_toc(toc_text)
        return self._parse_body(body_text, toc_entries)

    @staticmethod
    def _extract_raw_text(pdf_path: Path) -> str:
        # shared plumbing, same as IPC - see corpus/pdf_utils.py
        pages = extract_pdf_pages(pdf_path)
        pages = remove_repeated_headers(pages)
        return "\n".join(pages)

    @staticmethod
    def _split_toc_and_body(raw_text: str) -> tuple[str, str]:
        marker_pos = raw_text.find(BODY_START_MARKER)
        if marker_pos == -1:
            raise ValueError(
                f"couldn't find '{BODY_START_MARKER}' - BNS PDF layout may have changed, "
                f"check where the table of contents ends"
            )
        toc_text, body_text = raw_text[:marker_pos], raw_text[marker_pos:]

        end_pos = body_text.find(BODY_END_MARKER)
        if end_pos != -1:
            body_text = body_text[:end_pos]

        return toc_text, body_text

    @staticmethod
    def _clean_title(raw_title: str) -> str:
        """joins a (possibly page-break-wrapped) title into one line and
        drops any bare page-number line stuck to the end - see
        TOC_ENTRY's docstring for why this shows up at all."""
        lines = [line.strip() for line in raw_title.split("\n")]
        while lines and re.fullmatch(r"\d{1,4}", lines[-1] or ""):
            lines.pop()
        title = " ".join(line for line in lines if line)
        return title.rstrip(".").strip()

    @staticmethod
    def _parse_toc(toc_text: str) -> list[dict]:
        """returns ordered list of {"number", "title", "chapter", "is_stub"}."""
        chapters = list(CHAPTER_START.finditer(toc_text))
        entries = []

        for match in TOC_ENTRY.finditer(toc_text):
            number = match.group(1)
            title = BNSParser._clean_title(match.group(2))
            chapter = BNSParser._label_for_position(chapters, match.start())
            is_stub = title.lower().startswith(STUB_MARKERS)
            entries.append({"number": number, "title": title, "chapter": chapter, "is_stub": is_stub})

        return entries

    @staticmethod
    def _parse_body(body_text: str, toc_entries: list[dict]) -> list[Section]:
        chapters = list(CHAPTER_START.finditer(body_text))

        # same TOC-guided walk as IPC: search forward from an
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
            # the "N.—Title.—Body" sections (see BODY_CANDIDATE_TEMPLATE)
            # can leave a second, unconsumed dash character right at the
            # start of what we capture as body - strip it rather than
            # leave a stray leading "–" on the stored text.
            body = re.sub(r"^[\s\-\u2013\u2014]+", "", body).strip()

            chapter = BNSParser._label_for_position(chapters, match.start())
            metadata = {"chapter": chapter, "effective_date": EFFECTIVE_DATE}

            sections.append(Section(
                act=ACT, unit_type="section", number=entry["number"],
                title=entry["title"], body=body, status=DEFAULT_STATUS,
                metadata=metadata,
            ))

        # stub entries (Omitted/Repealed, if any exist in a later edition)
        # never had body text to find - record them factually from the
        # TOC's own bracketed text, nothing invented
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