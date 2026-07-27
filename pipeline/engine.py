"""
orchestrates the full error-detection pipeline for one document:
ML model + rule-based checkers -> merge -> deduplicate -> reading-order sort.

this is the only file that knows about model/ and rules/ together - those
two packages never import from each other, and neither imports pipeline/.
"""

from ocr.tokens import LineSpan
from model.schemas import ErrorSpan
from model.preprocess import build_chunks
from model.predict import predict
from model.postprocess import build_error_spans

from rules.registry import RULES

from pipeline.merger import merge_spans
from pipeline.deduplicate import deduplicate


def analyze(spans: list[LineSpan]) -> list[ErrorSpan]:
    ml_errors = _run_ml(spans)

    # run every registered rule checker — to add a new rule, edit rules/registry.py only.
    # each rule's own list of ErrorSpans is kept SEPARATE here (append, not
    # extend) - merge_spans(*span_lists) expects each argument to be one
    # source's own list, not a pre-flattened stream of individual ErrorSpan
    # objects. using .extend() here would flatten rule_errors into bare
    # ErrorSpan objects, which then get unpacked as individual (non-list)
    # arguments to merge_spans below - each one fails merger.py's own
    # `merged.extend(spans)` line, since a single ErrorSpan isn't iterable.
    rule_errors = []
    for rule in RULES:
        rule_errors.append(rule(spans))

    merged = merge_spans(ml_errors, *rule_errors)
    deduped = deduplicate(merged)

    return _sort_reading_order(deduped)


def _run_ml(spans: list[LineSpan]) -> list[ErrorSpan]:
    chunks = build_chunks(spans)
    label_id_sequences = predict(chunks)
    return build_error_spans(chunks, label_id_sequences, spans)


def _sort_reading_order(errors: list[ErrorSpan]) -> list[ErrorSpan]:
    """page by page, top-to-bottom, left-to-right - the order a human reading the document would hit them."""
    return sorted(errors, key=lambda e: (e.page_no, e.y0, e.x0))