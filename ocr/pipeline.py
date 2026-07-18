"""
single entrypoint for the entire OCR pipeline.
calls router which extracts native spans + identifies scanned pages
in one pass, then only calls surya on pages that actually need it.
"""

import re
from pathlib import Path
from typing import List, Optional

from ocr.tokens import LineSpan, sort_spans_by_reading_order
from ocr.router import route


def extract(
    pdf_path: Path, 
    min_chars_per_page: int = 2000,
    filter_noise: bool = True,
    detect_headings: bool = True,
) -> list[LineSpan]:
    """
    Extract all text from a PDF as LineSpans.
    
    Args:
        pdf_path: Path to the PDF file
        min_chars_per_page: Minimum characters to consider a page native
        filter_noise: Whether to filter out noise lines
        detect_headings: Whether to detect heading lines
        
    Returns:
        List of LineSpans in reading order
    """
    native_spans, scanned_pages = route(pdf_path, min_chars_per_page)
    
    spans = list(native_spans)
    
    if scanned_pages:
        # Lazy import so Surya models only load when actually needed
        from ocr.surya_extractor import SuryaExtractor
        surya = SuryaExtractor()
        spans.extend(surya.extract(pdf_path, scanned_pages))
    
    # Filter noise
    if filter_noise:
        spans = [s for s in spans if not s.is_noise()]
    
    # Sort by reading order
    spans = sort_spans_by_reading_order(spans)
    
    # Post-process: detect paragraph boundaries and headings
    spans = _detect_paragraphs(spans, detect_headings)
    
    return spans


def _detect_paragraphs(spans: list[LineSpan], detect_headings: bool = True) -> list[LineSpan]:
    """
    Detect paragraph boundaries in a list of spans.
    
    Sets is_paragraph_start and vertical_gap metadata.
    Uses adaptive threshold based on median gap.
    """
    if not spans:
        return spans
    
    # Group by page
    by_page: dict[int, list[LineSpan]] = {}
    for span in spans:
        by_page.setdefault(span.page_no, []).append(span)
    
    processed = []
    
    for page_no, page_spans in by_page.items():
        # Calculate vertical gaps
        gaps = []
        for i, span in enumerate(page_spans):
            if i == 0:
                span.vertical_gap = 0.0
            else:
                prev = page_spans[i - 1]
                gap = span.y0 - prev.y1
                span.vertical_gap = gap
                gaps.append(gap)
        
        # Calculate adaptive threshold using median gap
        if gaps:
            median_gap = sorted(gaps)[len(gaps) // 2]
            # Use 1.8x median gap as threshold (more robust than fixed multiplier)
            threshold = median_gap * 1.8
        else:
            threshold = 10.0  # fallback
        
        # Set paragraph starts
        for i, span in enumerate(page_spans):
            if i == 0:
                span.is_paragraph_start = True
            else:
                span.is_paragraph_start = span.vertical_gap > threshold
        
        # Detect headings using font metadata and text features
        if detect_headings:
            for span in page_spans:
                if _is_heading(span):
                    span.is_heading = True
        
        processed.extend(page_spans)
    
    return processed


def _is_heading(span: LineSpan) -> bool:
    """
    Detect if a line is a heading using multiple signals.
    
    Uses:
    - Font size relative to page average
    - Text length (short lines)
    - ALL CAPS
    - Ends with colon
    - Alphabetic ratio
    """
    text = span.text.strip()
    if not text:
        return False
    
    # Headings are usually short
    if len(text) > 100:
        return False
    
    # ALL CAPS is a strong indicator
    if text.isupper() and len(text) > 5:
        return True
    
    # Ends with colon
    if text.endswith(':'):
        return True
    
    # Short line with high alphabetic ratio (likely a heading)
    if len(text) < 60 and sum(c.isalpha() for c in text) / max(len(text), 1) > 0.8:
        return True
    
    # Font size signal (if available)
    if span.font_size:
        # Headings are usually larger than body text
        # We'll use the global average later
        pass
    
    return False