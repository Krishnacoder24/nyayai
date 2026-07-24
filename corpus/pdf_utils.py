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


def _format_superscripts(page: Page, baseline_size: float) -> Page:
    """returns a pdfplumber page with superscript footnote-marker digits
    replaced by bracketed digits (e.g., '1' becomes '[1]')."""
    if baseline_size <= 0:
        return page

    for obj in page.chars:
        if obj.get("object_type") == "char" and obj["text"].isdigit() and obj["size"] < baseline_size * SUPERSCRIPT_SIZE_RATIO:
            obj["text"] = f"[{obj['text']}]"

    return page


def extract_pdf_text(pdf_path: Path) -> str:
    """
    Extracts text from every page in reading order.
    Formats superscript footnote-marker digits with brackets.
    """
    with pdfplumber.open(pdf_path) as pdf:
        baseline = _dominant_font_size(pdf)
        pages = [_format_superscripts(page, baseline).extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


def extract_pdf_pages(pdf_path: Path) -> list[str]:
    """
    Returns one string per page.
    Useful when parsers need page-aware processing
    (e.g. removing TOC or headers). Formats superscript digits.
    """
    with pdfplumber.open(pdf_path) as pdf:
        baseline = _dominant_font_size(pdf)
        return [_format_superscripts(page, baseline).extract_text() or "" for page in pdf.pages]


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