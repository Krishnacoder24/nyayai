"""
decides per page whether to use native extraction or surya OCR.
uses multiple signals for better decision making.
"""

import pdfplumber
from pathlib import Path

from ocr.tokens import LineSpan
from ocr.native_extractor import NativeExtractor


def route(
    pdf_path: Path,
    min_chars_per_page: int = 20,
    min_lines_per_page: int = 3,
    min_alphabetic_ratio: float = 0.6,
    max_scanned_indicators: int = 1
) -> tuple[list[LineSpan], list[int]]:
    """
    Route pages to native extraction or OCR.
    
    Uses multiple signals to decide:
    - Character count
    - Line count
    - Alphabetic ratio (proportion of alphabetic characters)
    - Scanned document indicators
    
    Args:
        pdf_path: Path to the PDF
        min_chars_per_page: Minimum characters to consider text layer valid
        min_lines_per_page: Minimum lines to consider text layer valid
        min_alphabetic_ratio: Minimum alphabetic ratio to consider text layer valid
        max_scanned_indicators: Maximum scanned indicators to consider native
        
    Returns:
        Tuple of (native_spans, scanned_pages)
    """
    native = NativeExtractor()
    
    # Extract everything native can get in one pass
    all_spans, page_stats = native.extract(pdf_path)
    
    # Get total page count
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
    
    # Group spans by page
    spans_by_page: dict[int, list[LineSpan]] = {}
    for span in all_spans:
        spans_by_page.setdefault(span.page_no, []).append(span)
    
    native_spans = []
    scanned_pages = []
    
    for page_no in range(total_pages):
        # Get stats for this page
        stats = page_stats.get(page_no, {})
        
        # Decision logic using multiple signals
        is_native = (
            stats.get("char_count", 0) >= min_chars_per_page and
            stats.get("line_count", 0) >= min_lines_per_page and
            stats.get("alphabetic_ratio", 0) >= min_alphabetic_ratio and
            stats.get("scanned_indicators", 3) <= max_scanned_indicators
        )
        
        if is_native:
            native_spans.extend(spans_by_page.get(page_no, []))
        else:
            scanned_pages.append(page_no)
    
    return native_spans, scanned_pages