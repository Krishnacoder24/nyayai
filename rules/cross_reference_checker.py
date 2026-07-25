"""
checks internal cross-references within a document.

detects references to paragraphs, exhibits, and annexures (e.g. "as mentioned
in paragraph 3", "see Exhibit A") and verifies that the referenced item actually
exists somewhere in the document.

a "dangling" cross-reference — one that points to a paragraph/exhibit/annexure
that is never defined — is flagged as an error.

implementation is a placeholder for now.
actual logic to be implemented in Issue 39.
"""

from ocr.tokens import LineSpan
from model.schemas import ErrorSpan


def check_cross_references(spans: list[LineSpan]) -> list[ErrorSpan]:
    return []




