"""
draws colored highlight boxes on the original PDF, one per ErrorSpan, at
its exact bbox position, colored by error_type (see colors.py).

approach: reportlab can't edit an existing PDF's content stream directly,
so we build a one-page overlay (just the highlight rectangles, transparent
background) per original page that has errors, then merge it onto that
page with pypdf - the standard "watermark" trick.

coordinate systems differ between the two libraries, and getting this
wrong silently puts every highlight on the wrong half of the page:
  - ErrorSpan bboxes use pdfplumber's convention: origin top-left, y
    increases downward (see utils/bbox.py's docstring)
  - reportlab's canvas uses PDF's native convention: origin bottom-left,
    y increases upward
  so every y coordinate gets flipped against the page height before
  drawing.
"""

import io
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas as rl_canvas

from model.schemas import ErrorSpan
from renderer.colors import get_rgb
from config.constants import FILL_ALPHA, STROKE_ALPHA, STROKE_WIDTH



def annotate(pdf_path: Path, errors: list[ErrorSpan], output_path: Path) -> None:
    """reads pdf_path, draws every error's highlight box, writes the result to output_path."""
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()

    errors_by_page = _group_by_page(errors)

    for page_no, page in enumerate(reader.pages, start=1):
        page_errors = errors_by_page.get(page_no, [])
        if page_errors:
            width = float(page.mediabox.width)
            height = float(page.mediabox.height)
            overlay_page = _build_overlay_page(page_errors, width, height)
            page.merge_page(overlay_page) 
        writer.add_page(page)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        writer.write(f)


def _group_by_page(errors: list[ErrorSpan]) -> dict[int, list[ErrorSpan]]:
    grouped: dict[int, list[ErrorSpan]] = {}
    for error in errors:
        grouped.setdefault(error.page_no, []).append(error)
    return grouped


def _build_overlay_page(page_errors: list[ErrorSpan], width: float, height: float):
    buffer = io.BytesIO()
    c = rl_canvas.Canvas(buffer, pagesize=(width, height))

    for error in page_errors:
        r, g, b = get_rgb(error.error_type)
        x0, y0, x1, y1 = error.bbox

        # flip y: in pdfplumber space y0/y1 are measured down from the top.
        # reportlab's rect() wants the BOTTOM-left corner + a height going up,
        # so the rect's bottom edge is (page height - y1), not (page height - y0)
        rl_x = x0
        rl_y = height - y1
        rl_width = x1 - x0
        rl_height = y1 - y0

        if(rl_width <= 0 or rl_height <= 0):
            print(f"Warning: skipping invalid bbox for error '{error.text}' on page {error.page_no}: {error.bbox}")
            continue    
        

        c.saveState()
        c.setFillColorRGB(r, g, b, alpha=FILL_ALPHA)
        c.setStrokeColorRGB(r, g, b, alpha=STROKE_ALPHA)
        c.setLineWidth(STROKE_WIDTH)
        c.rect(rl_x, rl_y, rl_width, rl_height, fill=1, stroke=1)
        c.restoreState()

    c.save()
    buffer.seek(0)

    return PdfReader(buffer).pages[0]