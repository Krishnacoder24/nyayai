"""
regression test for issue #33: render_html() used to crash with
ValueError: Invalid format specifier on every report that had at least
one error, because _error_row()'s confidence column tried to put a
conditional inside an f-string's format spec. that's invalid Python, not
just a logic bug - so this needs to be caught before it ships again, not
just eyeballed.
"""

from pathlib import Path

from renderer.html_report import render_html


def _sample_report(errors: list[dict]) -> dict:
    errors_by_type: dict[str, int] = {}
    for e in errors:
        errors_by_type[e["error_type"]] = errors_by_type.get(e["error_type"], 0) + 1
    return {
        "source_filename": "sample.pdf",
        "total_errors": len(errors),
        "errors_by_type": errors_by_type,
        "errors": errors,
    }


def _base_error(**overrides) -> dict:
    error = {
        "text": "Sectoin 302",
        "error_type": "spelling",
        "page_no": 1,
        "bbox": [10.0, 20.0, 50.0, 30.0],
        "suggestion": "Section 302",
        "confidence": 0.87,
        "highlight_color": "#E63946",
    }
    error.update(overrides)
    return error


def test_render_html_with_a_real_confidence_value(tmp_path: Path):
    # this exact shape (a float confidence) is what was crashing every
    # single time before the fix - confidence defaults to 0.0 in
    # ErrorSpan, never actually None, so this branch was always hit.
    report = _sample_report([_base_error(confidence=0.87)])
    output_path = tmp_path / "report.html"

    render_html(report, output_path)

    html = output_path.read_text(encoding="utf-8")
    assert "0.87" in html
    assert "Sectoin 302" in html


def test_render_html_with_none_confidence(tmp_path: Path):
    # confidence isn't actually Optional in ErrorSpan today, but the
    # original code defended against None anyway - keep that working too.
    report = _sample_report([_base_error(confidence=None)])
    output_path = tmp_path / "report.html"

    render_html(report, output_path)

    html = output_path.read_text(encoding="utf-8")
    assert "—" in html


def test_render_html_with_multiple_error_types(tmp_path: Path):
    # one row per error type, so a bug isolated to a single _error_row()
    # call can't hide behind only ever testing one error.
    report = _sample_report(
        [
            _base_error(error_type="spelling", confidence=0.9),
            _base_error(error_type="grammar", confidence=0.5, text="a citation error"),
            _base_error(error_type="citation", confidence=None, text="Section 999 IPC"),
        ]
    )
    output_path = tmp_path / "report.html"

    render_html(report, output_path)

    html = output_path.read_text(encoding="utf-8")
    assert "0.90" in html
    assert "0.50" in html
    assert "—" in html


def test_render_html_with_no_errors(tmp_path: Path):
    report = _sample_report([])
    output_path = tmp_path / "report.html"

    render_html(report, output_path)

    html = output_path.read_text(encoding="utf-8")
    assert "No errors found." in html