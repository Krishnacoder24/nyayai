"""
tests for ocr.pipeline.extract().

sample_pdf_path (see tests/conftest.py) is a real, native-text PDF with
enough content per page to clear ocr/router.py's native-vs-scanned
thresholds (config.constants.MIN_CHARS_PER_PAGE etc.) - so this exercises
the real pdfplumber-based native extraction path without needing
surya-ocr, poppler, or a GPU. Surya's scanned-page path is out of scope
here; it needs an actual scanned (image-only) PDF fixture to mean
anything, which is a bigger fixture than this file wants to own - see
`make test-ocr FILE=...` for manually smoke-testing that path against a
real scanned FIR.
"""

from ocr.pipeline import extract
from ocr.tokens import LineSpan


def test_extract_returns_one_linespan_per_line(sample_pdf_path):
    spans = extract(sample_pdf_path)

    assert len(spans) == 7
    assert all(isinstance(s, LineSpan) for s in spans)


def test_extract_uses_native_source_for_a_text_pdf(sample_pdf_path):
    spans = extract(sample_pdf_path)

    assert {s.source for s in spans} == {"native"}


def test_extract_preserves_line_text_and_order(sample_pdf_path):
    spans = extract(sample_pdf_path)

    texts = [s.text for s in spans]
    assert texts[0] == "FIRST INFORMATION REPORT"
    assert texts[1] == "Police Station: Kotwali, District: Patna"
    assert "Section 302 IPC" in texts[3]
    assert "Section 103 BNS" in texts[3]
    assert "Ramesh Kumar" in texts[4]
    assert "Rakesh Kumar" in texts[5]


def test_extract_gives_every_span_a_valid_bbox_on_page_zero(sample_pdf_path):
    spans = extract(sample_pdf_path)

    for span in spans:
        assert span.page_no == 0
        x0, y0, x1, y1 = span.bbox
        # top-left origin, y-down (pdfplumber convention) - x1 > x0
        # and y1 > y0 for every real line, never a degenerate/zero box
        assert x1 > x0
        assert y1 > y0


def test_extract_reports_spans_in_top_to_bottom_reading_order(sample_pdf_path):
    spans = extract(sample_pdf_path)

    y_positions = [s.y0 for s in spans]
    assert y_positions == sorted(y_positions)


def test_extract_filters_page_number_noise(tmp_path, sample_pdf_bytes):
    # is_noise() (ocr/tokens.py) already covers "Page 3" / bare "12" /
    # separator lines - append one and confirm extract() actually drops
    # it via filter_noise=True (the default), rather than just trusting
    # LineSpan.is_noise()'s own unit-level behaviour.
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4

    path = tmp_path / "with_page_number.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setFont("Helvetica", 12)
    lines = [
        "FIRST INFORMATION REPORT",
        "Police Station: Kotwali, District: Patna",
        "FIR No: 145/2024",
        "Under Section 302 IPC and Section 103 BNS",
        "Complainant: Ramesh Kumar, S/O Late Suresh Kumar",
        "The complainant Rakesh Kumar stated that on the night of the incident,",
        "he witnessed the accused near the scene as described in paragraph 1.",
        "Page 1",
    ]
    y = 800
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.showPage()
    c.save()

    spans = extract(path)

    assert "Page 1" not in [s.text for s in spans]