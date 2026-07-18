"""
extracts text from scanned/image-only pdfs using surya OCR.

surya natively gives line-level text + line-level bboxes which maps
directly to LineSpan - no approximation needed anymore. this is the
whole reason we switched from WordToken to LineSpan.

Improvements:
- Confidence tracking for each line
- Layout detection for paragraphs/headings
- Better noise filtering
- Configurable DPI for accuracy/performance tradeoff
- Batch processing with memory optimization
"""

import pypdfium2 as pdfium
import re
from PIL import Image
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor

from ocr.tokens import LineSpan


class SuryaExtractor:
    """
    Extract text from scanned pages using Surya OCR.
    
    Features:
    - Configurable DPI (144-300)
    - Batch processing to manage VRAM
    - Confidence scores per line
    - Layout analysis for paragraph boundaries
    - Noise filtering
    """
    
    # Default rendering scale (2.0 = ~144 DPI)
    # Surya accuracy improves up to ~200 DPI before diminishing returns
    DEFAULT_SCALE = 2.0  # 144 DPI
    HIGH_ACCURACY_SCALE = 2.8  # ~200 DPI
    
    def __init__(
        self, 
        scale: float = DEFAULT_SCALE,
        chunk_size: int = 4,
        min_confidence: float = 0.3,
        detect_layout: bool = True,
    ):
        """
        Initialize Surya extractor.
        
        Args:
            scale: Rendering scale for PDF pages (2.0 = 144 DPI, 2.8 = 200 DPI)
            chunk_size: Number of pages to process at once (VRAM constraint)
            min_confidence: Minimum confidence threshold for lines (0-1)
            detect_layout: Whether to run layout analysis for paragraph detection
        """
        self.scale = scale
        self.chunk_size = chunk_size
        self.min_confidence = min_confidence
        self.detect_layout = detect_layout
        
        # Load models once, reuse across all pages
        # Don't reinstantiate per page
        self.detection_predictor = DetectionPredictor()
        self.recognition_predictor = RecognitionPredictor()
        
        # Cache for rendered pages to avoid re-rendering
        self._image_cache: Dict[int, Image.Image] = {}
    
    def _render_pages(
        self, 
        pdf_path: Path, 
        page_numbers: List[int]
    ) -> List[Image.Image]:
        """
        Render requested pages as PIL Images.
        
        Opens the PDF once, renders all requested pages, closes it.
        One parse of the PDF header instead of one per page.
        
        Args:
            pdf_path: Path to the PDF
            page_numbers: List of page numbers (0-indexed)
            
        Returns:
            List of PIL Images
        """
        images = []
        doc = pdfium.PdfDocument(pdf_path)
        
        try:
            for page_no in page_numbers:
                # Check cache first
                if page_no in self._image_cache:
                    images.append(self._image_cache[page_no])
                    continue
                
                bitmap = doc[page_no].render(scale=self.scale) #type: ignore
                img = bitmap.to_pil()
                self._image_cache[page_no] = img
                images.append(img)
        finally:
            doc.close()
        
        return images
    
    def _filter_noise_lines(self, lines: List[Any], page_no: int) -> List[Tuple[str, Tuple[float, float, float, float], Optional[float]]]:
        """
        Filter out noise lines from Surya output.
        
        Returns list of (text, bbox, confidence) tuples.
        """
        filtered = []
        
        for line in lines:
            text = line.text.strip()
            if not text:
                continue
            
            # Skip very short lines (likely noise)
            if len(text) < 2:
                continue
            
            # Skip lines that are just separators
            if all(c in "-_=~*" for c in text):
                continue
            
            # Skip page numbers
            if text.isdigit():
                continue
            if re.match(r'^Page \d+$', text, re.I):
                continue
            
            # Skip URLs
            if re.match(r'^www\.[a-z0-9]+\.[a-z]+$', text, re.I):
                continue
            if re.match(r'^https?://', text, re.I):
                continue
            
            # Get confidence
            confidence = None
            if hasattr(line, 'confidence'):
                confidence = line.confidence
            
            # Skip low confidence lines
            if confidence is not None and confidence < self.min_confidence:
                continue
            
            # Get bbox
            bbox = line.bbox
            
            # Skip invalid bboxes
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            
            filtered.append((text, bbox, confidence))
        
        return filtered
    
    def _detect_layout(self, lines: List[Tuple[str, Tuple[float, float, float, float], Optional[float]]]) -> List[Dict[str, Any]]:
        """
        Detect layout information from lines.
        
        Returns list of dicts with layout metadata for each line.
        """
        if not lines:
            return []
        
        # Sort by vertical position
        sorted_lines = sorted(lines, key=lambda x: x[1][1])
        
        result = []
        prev_y1 = None
        gaps = []
        
        # First pass: calculate gaps
        for i, (text, bbox, confidence) in enumerate(sorted_lines):
            x0, y0, x1, y1 = bbox
            if i == 0:
                gap = 0.0
            else:
                _, prev_bbox, _ = sorted_lines[i-1]
                gap = y0 - prev_bbox[3]
                gaps.append(gap)
            
            result.append({
                "text": text,
                "bbox": bbox,
                "confidence": confidence,
                "gap": gap,
                "is_heading": False,
                "is_paragraph_start": False,
            })
        
        # Calculate adaptive threshold
        if gaps:
            median_gap = sorted(gaps)[len(gaps) // 2]
            threshold = median_gap * 1.8
        else:
            threshold = 10.0
        
        # Second pass: detect paragraph starts and headings
        for i, item in enumerate(result):
            # Paragraph start
            if i == 0:
                item["is_paragraph_start"] = True
            else:
                item["is_paragraph_start"] = item["gap"] > threshold
            
            # Heading detection
            text = item["text"]
            if len(text) < 80:
                # ALL CAPS
                if text.isupper() and len(text) > 5:
                    item["is_heading"] = True
                # Ends with colon
                elif text.endswith(':'):
                    item["is_heading"] = True
                # Short line with high alphabetic ratio
                elif len(text) < 60 and sum(c.isalpha() for c in text) / max(len(text), 1) > 0.8:
                    item["is_heading"] = True
        
        return result
    
    def extract(
        self, 
        pdf_path: Path, 
        page_numbers: List[int]
    ) -> List[LineSpan]:
        """
        Run Surya OCR on the given page numbers.
        
        Renders all pages with one PDF open, then feeds Surya in chunks
        to avoid OOM on large scanned documents.
        
        Args:
            pdf_path: Path to the PDF
            page_numbers: List of page numbers (0-indexed)
            
        Returns:
            List of LineSpan objects
        """
        if not page_numbers:
            return []
        
        # Render all pages
        images = self._render_pages(pdf_path, page_numbers)
        
        all_spans = []
        
        # Process in chunks
        for i in range(0, len(images), self.chunk_size):
            chunk_images = images[i:i + self.chunk_size]
            chunk_page_nos = page_numbers[i:i + self.chunk_size]
            
            # Run detection + recognition
            try:
                results = self.recognition_predictor(
                    chunk_images,
                    [None] * len(chunk_images),
                    self.detection_predictor,
                )
            except Exception as e:
                print(f"Error processing pages {chunk_page_nos}: {e}")
                continue
            
            for page_no, page_result in zip(chunk_page_nos, results):
                # Filter noise lines
                filtered = self._filter_noise_lines(
                    page_result.text_lines, 
                    page_no
                )
                
                if not filtered:
                    continue
                
                # Detect layout if enabled
                if self.detect_layout:
                    layout_info = self._detect_layout(filtered)
                else:
                    layout_info = [{"is_heading": False, "is_paragraph_start": False} 
                                  for _ in filtered]
                
                # Create LineSpans
                for line_info, layout in zip(filtered, layout_info):
                    text, bbox, confidence = line_info
                    x0, y0, x1, y1 = bbox
                    
                    span = LineSpan(
                        text=text,
                        page_no=page_no,
                        source="surya",
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        confidence=confidence,
                        is_heading=layout.get("is_heading", False),
                        is_paragraph_start=layout.get("is_paragraph_start", False),
                        vertical_gap=layout.get("gap", 0.0),
                    )
                    
                    if span.is_valid():
                        all_spans.append(span)
        
        return all_spans
    
    def extract_with_stats(
        self, 
        pdf_path: Path, 
        page_numbers: List[int]
    ) -> Tuple[List[LineSpan], Dict[int, Dict[str, Any]]]:
        """
        Extract text and return page statistics.
        
        Returns:
            Tuple of (spans, page_stats)
        """
        spans = self.extract(pdf_path, page_numbers)
        
        # Group spans by page
        by_page: Dict[int, List[LineSpan]] = {}
        for span in spans:
            by_page.setdefault(span.page_no, []).append(span)
        
        # Calculate stats per page
        page_stats = {}
        for page_no, page_spans in by_page.items():
            page_stats[page_no] = {
                "line_count": len(page_spans),
                "char_count": sum(len(s.text) for s in page_spans),
                "avg_confidence": sum(s.confidence or 0 for s in page_spans) / max(len(page_spans), 1),
                "heading_count": sum(1 for s in page_spans if s.is_heading),
            }
        
        return spans, page_stats
    
    def clear_cache(self):
        """Clear the image cache to free memory."""
        self._image_cache.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_cache()


# Convenience function for one-off extraction
def extract_scanned(
    pdf_path: Path,
    page_numbers: List[int],
    scale: float = SuryaExtractor.DEFAULT_SCALE,
    chunk_size: int = 4,
    min_confidence: float = 0.3,
    detect_layout: bool = True,
) -> List[LineSpan]:
    """
    Convenience function to extract text from scanned pages.
    
    Args:
        pdf_path: Path to the PDF
        page_numbers: List of page numbers (0-indexed)
        scale: Rendering scale (2.0 = 144 DPI)
        chunk_size: Number of pages to process at once
        min_confidence: Minimum confidence threshold
        detect_layout: Whether to detect layout
        
    Returns:
        List of LineSpans
    """
    with SuryaExtractor(
        scale=scale,
        chunk_size=chunk_size,
        min_confidence=min_confidence,
        detect_layout=detect_layout,
    ) as extractor:
        return extractor.extract(pdf_path, page_numbers)