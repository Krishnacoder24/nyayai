"""
builds the structured report: a plain, JSON-serializable summary of every
ErrorSpan found in a document. this is what api/routes returns alongside
the annotated PDF, and what html_report.py turns into a standalone page.
"""

from dataclasses import asdict

from model.schemas import ErrorSpan


def build_report(errors: list[ErrorSpan], source_filename: str = "") -> dict:
    errors_by_type: dict[str, int] = {}
    for error in errors:
        errors_by_type[error.error_type] = errors_by_type.get(error.error_type, 0) + 1

    return {
        "source_filename": source_filename,
        "total_errors": len(errors),
        "errors_by_type": errors_by_type,
        "errors": [_error_to_dict(e) for e in errors],
    }


def _error_to_dict(error: ErrorSpan) -> dict:
    # asdict() only pulls the dataclass's real fields - bbox and
    # highlight_color are @property, so they're added in separately
    data = asdict(error)
    data["bbox"] = list(error.bbox)
    data["highlight_color"] = error.highlight_color
    return data