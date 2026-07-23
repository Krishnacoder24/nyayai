from pathlib import Path
import statistics

import pdfplumber
from pdfplumber.page import Page

# footnote reference markers (like the digit after "date" that marks a
# footnote at the bottom of the page) are set in a genuinely smaller font
# in the source PDF - not just a plain digit glued onto a word by
# coincidence. confirmed directly against bns.pdf's character data: the
# "1" in "...on such date1 as the Central Government..." has size 6.96
# against a page body-text size of 11.04, vs. a real "(1)" elsewhere on
# the same page sitting at the full 11.04. left unfiltered, "date" and
# "date1" embed as different tokens for what's semantically the same
# word plus an unrelated footnote marker - this ratio check catches that
# reliably by comparing each digit's actual rendered size against the
# page's own body-text baseline, rather than guessing from the plain
# text string (which can't tell "date1" apart from a real word ending in
# a digit without risking false positives on legitimate content).
SUPERSCRIPT_SIZE_RATIO = 0.85


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
    import re

    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()