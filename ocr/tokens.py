"""
LineSpan represents a single line of text extracted from a PDF page.
Includes metadata for paragraph detection, reading order, and confidence.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class LineSpan:
    """A single line of text with its bounding box and metadata."""
    
    # Core fields
    text: str                                      # full line text, words joined with single space
    page_no: int                                   # 0-indexed page number
    source: str                                    # "native" or "surya"
    
    # Bounding box
    x0: float                                      # left edge
    y0: float                                      # top edge
    x1: float                                      # right edge
    y1: float                                      # bottom edge
    
    # Optional metadata fields (populated when available)
    line_no: Optional[int] = None                  # line number within page
    confidence: Optional[float] = None             # OCR confidence (0-1)
    
    
    # Paragraph detection fields
    is_heading: bool = False                       # line appears to be a heading
    is_paragraph_start: bool = False               # line starts a new paragraph
    vertical_gap: float = 0.0                      # gap above this line in pts
    
    # Font metadata (when available from native extraction)
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    
    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return bounding box as (x0, y0, x1, y1)."""
        return (self.x0, self.y0, self.x1, self.y1)
    
    @property
    def width(self) -> float:
        """Line width in points."""
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        """Line height in points."""
        return self.y1 - self.y0
    
    @property
    def center_x(self) -> float:
        """Horizontal center of the line."""
        return (self.x0 + self.x1) / 2
    
    @property
    def center_y(self) -> float:
        """Vertical center of the line."""
        return (self.y0 + self.y1) / 2
    
    def is_valid(self) -> bool:
        """Check if this span has valid text and geometry."""
        if not self.text or not self.text.strip():
            return False
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            return False
        # Reject very short lines that are probably noise
        if len(self.text.strip()) < 2:
            return False
        return True
    
    def is_noise(self) -> bool:
        """Check if this line is likely noise (page numbers, separators, etc.)."""
        text = self.text.strip()
        if not text:
            return True
        
        # Reject lines that are just separators
        if all(c in "-_=~*" for c in text):
            return True
        
        # Reject page numbers
        if re.match(r'^Page \d+$', text, re.I):
            return True
        if re.match(r'^\d+$', text):
            return True
        if re.match(r'^\d+ of \d+$', text):
            return True
        
        # Reject URLs
        if re.match(r'^www\.[a-z0-9]+\.[a-z]+$', text, re.I):
            return True
        if re.match(r'^https?://', text, re.I):
            return True
        
        # Reject very short all-caps lines (often headers/footers)
        if len(text) < 15 and text.isupper():
            return True
        
        # Reject lines with only punctuation/dots
        if all(c in ".,;:!?()[]{}" or c.isspace() for c in text):
            return True
        
        # Reject lines that are just bullet points
        if re.match(r'^[•·●○■□▪▫]+\s*$', text):
            return True
        
        return False
    
    def __repr__(self) -> str:
        return f"LineSpan(page={self.page_no}, text='{self.text[:50]}...')"


def sort_spans_by_reading_order(spans: list[LineSpan]) -> list[LineSpan]:
    """
    Sort spans by page, then by vertical position (top-to-bottom),
    with horizontal tie-breaking for multi-column layouts.
    
    This is a simple implementation. For complex multi-column layouts,
    a full layout analysis is needed.
    """
    def sort_key(s: LineSpan) -> tuple[int, float, float]:
        # Primary: page number
        # Secondary: y0 (top position)
        # Tertiary: x0 (left-to-right for same y)
        return (s.page_no, s.y0, s.x0)
    
    return sorted(spans, key=sort_key)