"""
checks internal cross-references within a document.

detects references to paragraphs, exhibits, and annexures (e.g. "as mentioned
in paragraph 3", "see Exhibit A") and verifies that the referenced item actually
exists somewhere in the document.

a "dangling" cross-reference — one that points to a paragraph/exhibit/annexure
that is never defined — is flagged as an error.

"""

import re
import logging

from ocr.tokens import LineSpan
from model.schemas import ErrorSpan

logger = logging.getLogger(__name__)


# REGEX PATTERNS - DEFINITIONS (what actually exists in the document)

# matches lines that define a numbered paragraph
# e.g. "1. The accused was present..." or "3. The witness stated..."
DEFINITION_PARAGRAPH = re.compile(
    r'^\s*(\d+)\.\s+[A-Za-z]',
    re.MULTILINE,
)

# matches lines that define an exhibit
# e.g. "Exhibit A:" or "Exhibit B —" or "EXHIBIT C."
DEFINITION_EXHIBIT = re.compile(
    r'\bExhibit\s+([A-Z]{1,3})\b',
    re.IGNORECASE,
)

# matches lines that define an annexure
# e.g. "Annexure I —" or "Annexure II:" or "ANNEXURE III"
DEFINITION_ANNEXURE = re.compile(
    r'\bAnnexure\s+([IVXLCDM]+)\b',
    re.IGNORECASE,
)



# REGEX PATTERNS - REFERENCES (what the document points to)

# e.g. "as mentioned in paragraph 3", "refer to paragraph 5", "see paragraph 2"
REFERENCE_PARAGRAPH = re.compile(
    r'\bparagraph\s+(\d+)\b',
    re.IGNORECASE,
)

# e.g. "see Exhibit A", "refer Exhibit B", "as per Exhibit C"
REFERENCE_EXHIBIT = re.compile(
    r'\bExhibit\s+([A-Z]{1,3})\b',
    re.IGNORECASE,
)

# e.g. "refer to Annexure I", "as per Annexure II"
REFERENCE_ANNEXURE = re.compile(
    r'\bAnnexure\s+([IVXLCDM]+)\b',
    re.IGNORECASE,
)

# words that tell us a line is pointing to something, not defining it
# e.g. "as mentioned in paragraph 3" — "mentioned in" is the signal here
REFERENCE_SIGNALS = re.compile(
    r'\b(mentioned in|refer(?:red)? to|see|as per|pursuant to|stated in|per)\b',
    re.IGNORECASE,
)


def check_cross_references(spans: list[LineSpan]) -> list[ErrorSpan]:
    """
    detects dangling cross-references in a document using a single loop.

    single loop approach:
      - iterate over all spans once, collecting definitions AND references
        into two separate buckets simultaneously
      - after the loop ends (full document scanned), compare references
        against definitions
      - any reference whose target is not in definitions → dangling → ErrorSpan
    """
    # single loop: collect definitions and references together
    definitions: set[str] = set()   # normalised keys of what actually EXISTS
    references: list[dict] = []     # each dict: {target, span, matched_text}

    for span in spans:
        _collect_definitions(span, definitions)
        _collect_references(span, references)

    logger.debug(f"cross_reference_checker: {len(definitions)} definitions, {len(references)} references found")

    # compare after loop: full document scanned, safe to check now
    errors: list[ErrorSpan] = []

    for ref in references:
        if ref["target"] not in definitions:   # referenced but never defined → flag it
            errors.append(ErrorSpan(
                text=ref["matched_text"],
                error_type="cross_reference",
                page_no=ref["span"].page_no,
                x0=ref["span"].x0,
                y0=ref["span"].y0,
                x1=ref["span"].x1,
                y1=ref["span"].y1,
                suggestion=f'"{ref["target"]}" is referenced but never defined in this document',
                confidence=0.90,
                source="cross_reference_rule",
            ))

    return errors


# PRIVATE HELPERS

def _collect_definitions(span: LineSpan, definitions: set) -> None:
    """
    regex scan: find lines that DEFINE a paragraph/exhibit/annexure.
    adds normalised keys (e.g. "paragraph 1", "exhibit a", "annexure ii")
    to the definitions set.

    definition signals:
      - numbered paragraphs: "1. The accused..." at the start of a line
      - exhibits: "Exhibit A:" or "Exhibit B —"
      - annexures: "Annexure I —" or "Annexure II:"
    """
    text = span.text

    # numbered paragraphs — "1. Some text" at start of a line is a definition
    for match in DEFINITION_PARAGRAPH.finditer(text):
        number = match.group(1)
        definitions.add(f"paragraph {number}")

    # exhibits — "Exhibit A" as a heading (not after a reference signal word)
    for match in DEFINITION_EXHIBIT.finditer(text):
        # only treat it as a definition if there's no reference signal before it on the same line
        line_before = text[:match.start()].rsplit("\n", 1)[-1]
        if not REFERENCE_SIGNALS.search(line_before):
            label = match.group(1).lower()
            definitions.add(f"exhibit {label}")

    # annexures — "Annexure I" as a heading
    for match in DEFINITION_ANNEXURE.finditer(text):
        line_before = text[:match.start()].rsplit("\n", 1)[-1]
        if not REFERENCE_SIGNALS.search(line_before):
            label = match.group(1).lower()
            definitions.add(f"annexure {label}")


def _collect_references(span: LineSpan, references: list) -> None:
    """
    regex scan: find lines that POINT TO a paragraph/exhibit/annexure.
    only collects matches that appear after a reference signal word
    (e.g. "as mentioned in", "see", "refer to") to avoid false positives
    from definition lines.

    appends dicts of shape:
      {"target": "paragraph 3", "span": span, "matched_text": "paragraph 3"}
    """
    text = span.text

    # paragraph references — only after a signal word
    for match in REFERENCE_PARAGRAPH.finditer(text):
        line_before = text[:match.start()].rsplit("\n", 1)[-1]
        if REFERENCE_SIGNALS.search(line_before) or _is_reference_context(text, match.start()):
            number = match.group(1)
            references.append({
                "target": f"paragraph {number}",
                "span": span,
                "matched_text": match.group(0),
            })

    # exhibit references — only after a signal word
    for match in REFERENCE_EXHIBIT.finditer(text):
        line_before = text[:match.start()].rsplit("\n", 1)[-1]
        if REFERENCE_SIGNALS.search(line_before):
            label = match.group(1).lower()
            references.append({
                "target": f"exhibit {label}",
                "span": span,
                "matched_text": match.group(0),
            })

    # annexure references — only after a signal word
    for match in REFERENCE_ANNEXURE.finditer(text):
        line_before = text[:match.start()].rsplit("\n", 1)[-1]
        if REFERENCE_SIGNALS.search(line_before):
            label = match.group(1).lower()
            references.append({
                "target": f"annexure {label}",
                "span": span,
                "matched_text": match.group(0),
            })


def _is_reference_context(text: str, match_pos: int) -> bool:
    """
    checks a small window of text just before the match position
    for reference signal words, in case the signal is slightly earlier
    on the same line but not captured by rsplit.
    """
    window_start = max(0, match_pos - 60)
    window = text[window_start:match_pos]
    return bool(REFERENCE_SIGNALS.search(window))
