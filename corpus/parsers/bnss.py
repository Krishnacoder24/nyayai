"""
BNSS parser. fully self-contained - no shared base class or inheritance
with any other act's parser (see issue #26). BNSS is a new act replacing
the CrPC (2023, in force from 2024).

page extraction and running-header stripping are shared plumbing, not
act-specific grammar, so they still come from corpus/pdf_utils.py rather
than being reimplemented here.

approach: same TOC-guided idea as IPC/BNS - walk the TOC in order, and for
each expected number, search FORWARD from a monotonically-advancing
cursor for a candidate matching that exact number.

footnote shape: corpus/pdf_utils.py's _format_superscripts() wraps any
digit rendered at superscript size in its own "[...]" before this parser
ever sees the text (verified against real character-size data - see that
file's docstring). so a footnote-marked amendment/insertion comes through
as "[N][body...]" - a bracketed footnote number immediately followed by
the real opening bracket - not a bare digit glued onto the bracket. the
optional prefixes below are written to match that shape.
"""

import re
from pathlib import Path

from corpus.schemas import Section
from corpus.pdf_utils import extract_pdf_pages, remove_repeated_headers

ACT = "BNSS"
DEFAULT_STATUS = "active"

# BNSS came into force on 1st July 2024
EFFECTIVE_DATE = "2024-07-01"

# marks where the TOC ends and the actual numbered Act text begins.
BODY_START_MARKER = "BE it enacted by Parliament"

# after section 531 the PDF continues into The First Schedule.
BODY_END_MARKER = "THE FIRST SCHEDULE"

# TOC entries
TOC_ENTRY = re.compile(
    r'\n(\d{1,3})\.\s+([\s\S]+?)'
    r'(?=\n\d{1,3}\.\s|\nCHAPTER\s|\n[A-Z][a-z][^\n]*\n\d|\n\d{1,4}\n|\Z)'
)

# chapter headers, in both TOC and body. optional prefix is a bracketed
# footnote number followed by the real opening bracket ("[N]["), matching
# pdf_utils._format_superscripts()'s output - see top docstring.
CHAPTER_START = re.compile(r'\n\s*(?:\[\d{1,3}\]\s*\[)?\s*CHAPTER\s+([IVXLCDM]+[A-Z]?)\s*\n\s*([^\n]*)')

# optional prefix is a bracketed footnote number followed by the real
# opening bracket ("[N][") - matches pdf_utils._format_superscripts()'s
# output, not a bare digit glued onto the bracket. see top docstring.
BODY_CANDIDATE_TEMPLATE = (
    r'(?:^|\n)\s*(?:\[\d{{1,3}}\]\s*\[|\[)?\s*{number}(?![A-Za-z0-9])[\s.]{{1,3}}[-\u2013\u2014]?\s*'
    r'(?:[A-Za-z"\u2018\u201c][\s\S]{{0,250}}?)\.?\s*[-\u2013\u2014]'
)


def _candidate_pattern(number: str) -> re.Pattern:
    return re.compile(BODY_CANDIDATE_TEMPLATE.format(number=re.escape(number)), re.MULTILINE)


STUB_MARKERS = ("[omitted", "[repealed")


class BNSSParser:
    act = ACT

    def parse(self, pdf_path: Path) -> list[Section]:
        raw_text = self._extract_raw_text(pdf_path)
        toc_text, body_text = self._split_toc_and_body(raw_text)
        toc_entries = self._parse_toc(toc_text)
        return self._parse_body(body_text, toc_entries)

    @staticmethod
    def _extract_raw_text(pdf_path: Path) -> str:
        pages = extract_pdf_pages(pdf_path)
        pages = remove_repeated_headers(pages)
        return "\n".join(pages)

    @staticmethod
    def _split_toc_and_body(raw_text: str) -> tuple[str, str]:
        marker_pos = raw_text.find(BODY_START_MARKER)
        if marker_pos == -1:
            raise ValueError(
                f"couldn't find '{BODY_START_MARKER}' - BNSS PDF layout may have changed, "
                f"check where the table of contents ends"
            )
        toc_text, body_text = raw_text[:marker_pos], raw_text[marker_pos:]

        end_pos = body_text.find(BODY_END_MARKER)
        if end_pos != -1:
            body_text = body_text[:end_pos]

        return toc_text, body_text

    @staticmethod
    def _clean_title(raw_title: str) -> str:
        lines = [line.strip() for line in raw_title.split("\n")]
        while lines and re.fullmatch(r"\d{1,4}", lines[-1] or ""):
            lines.pop()
        title = " ".join(line for line in lines if line)
        
        # BNSS specific: the last TOC entry (531) grabs "THE FIRST SCHEDULE" as part of its title 
        if "THE FIRST SCHEDULE" in title:
            title = title.split("THE FIRST SCHEDULE")[0].strip()
            
        return title.rstrip(".").strip()

    @staticmethod
    def _parse_toc(toc_text: str) -> list[dict]:
        chapters = list(CHAPTER_START.finditer(toc_text))
        entries = []

        for match in TOC_ENTRY.finditer(toc_text):
            number = match.group(1)
            title = BNSSParser._clean_title(match.group(2))
            chapter = BNSSParser._label_for_position(chapters, match.start())
            is_stub = title.lower().startswith(STUB_MARKERS)
            entries.append({"number": number, "title": title, "chapter": chapter, "is_stub": is_stub})

        return entries

    @staticmethod
    def _parse_body(body_text: str, toc_entries: list[dict]) -> list[Section]:
        chapters = list(CHAPTER_START.finditer(body_text))

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
            body = re.sub(r"^[\s\-\u2013\u2014]+", "", body).strip()

            chapter = BNSSParser._label_for_position(chapters, match.start())
            metadata = {"chapter": chapter, "effective_date": EFFECTIVE_DATE}

            sections.append(Section(
                act=ACT, unit_type="section", number=entry["number"],
                title=entry["title"], body=body, status=DEFAULT_STATUS,
                metadata=metadata,
            ))

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