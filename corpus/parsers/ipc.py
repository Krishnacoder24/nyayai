"""
IPC parser. fully self-contained - no shared base class with any other
act's parser. IPC's real PDF has real-world noise that a generic parser
would fight, and none of it is guaranteed to look anything like BNS/BNSS's
(IPC is a 160-year-old, heavily-amended act; BNS/BNSS are brand new).

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

import pdfplumber

from corpus.schemas import Section
from corpus.data.ipc_bns_mapping import IPC_TO_BNS

ACT = "IPC"
DEFAULT_STATUS = "repealed"  # BNS replaced the IPC in full, effective 2024-07-01
EFFECTIVE_DATE = "1860-01-01"

# marks where the TOC ends and the actual numbered Act text begins
BODY_START_MARKER = "ACT NO. 45 OF 1860"

# TOC lines: "   9.   Number.\n" or "   10. "Man". "Woman".\n" - no dash, ends at newline
TOC_ENTRY = re.compile(r'\n\s*(\d{1,3}[A-Z]{0,2})\.\s+(.+)')

# chapter headers, in both TOC and body: "CHAPTER XVI" / "CHAPTER VA" (roman + optional letter)
CHAPTER_START = re.compile(r'\n\s*CHAPTER\s+([IVXLCDM]+[A-Z]?)\s*\n\s*([^\n]*)')

# candidate section start in the body. the optional footnote-digit +
# bracket combo has to be ONE atomic optional unit, not two independently
# optional pieces - tried that first and it backfired: for a plain
# unbracketed "10. "Man"..." section, a separately-optional greedy digit
# prefix happily "matched" the "1" of "10" as if it were a footnote
# marker, leaving only "0" behind for the real number capture. requiring
# the digit prefix to be followed by an actual "[" (or just "[" alone,
# for bracketed sections with no footnote marker in front) closes that
# gap - confirmed by testing both the "10." and "9[4." cases directly.
# separator after the real number requires at least one space/period char
# - without a minimum, "6th October" in the preamble date matched as a
# false candidate. the trailing ".—" requirement excludes footnote-list
# prose ("1. The Indian Penal Code has been extended...") - real
# "N. Sentence." prose, but never followed by a dash right after a short
# title, unlike a real section header.
BODY_CANDIDATE = re.compile(
    r'(?:^|\n)\s*(?:\d{1,3}\s*\[|\[)?\s*(\d{1,3}[A-Z]{0,2})[\s.]{1,3}(?:[A-Za-z"\u2018\u201c][\s\S]{0,250}?)\.\s*[-\u2013\u2014]',
    re.MULTILINE,
)

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
            pages = [page.extract_text() or "" for page in pdf.pages]
        pages = IPCParser._strip_running_headers(pages)
        return "\n".join(pages)

    @staticmethod
    def _strip_running_headers(pages: list[str]) -> list[str]:
        """a line repeated across most pages (act name, page number) is boilerplate, not section text."""
        if len(pages) < 3:
            return pages

        line_counts: dict[str, int] = {}
        for page in pages:
            for line in page.splitlines():
                line = line.strip()
                if line:
                    line_counts[line] = line_counts.get(line, 0) + 1

        threshold = len(pages) * 0.5
        boilerplate = {line for line, count in line_counts.items() if count >= threshold}

        cleaned = []
        for page in pages:
            kept = [ln for ln in page.splitlines() if ln.strip() not in boilerplate]
            cleaned.append("\n".join(kept))
        return cleaned

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
        candidates = list(BODY_CANDIDATE.finditer(body_text))

        # pass 1: walk candidates in document order, consuming exactly one
        # match per expected (non-stub) TOC number, in TOC order. anything
        # that doesn't match the currently-expected number is noise (a
        # footnote marker, a stray number inside running text) and is skipped.
        matched: dict[int, re.Match] = {}
        candidate_idx = 0
        for i, entry in enumerate(toc_entries):
            if entry["is_stub"]:
                continue
            while candidate_idx < len(candidates):
                cand = candidates[candidate_idx]
                candidate_idx += 1
                if cand.group(1) == entry["number"]:
                    matched[i] = cand
                    break

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