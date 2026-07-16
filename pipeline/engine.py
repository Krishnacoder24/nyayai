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

from rules.citation_checker import check_citations
from rules.entity_checker import check_entities

from pipeline.merger import merge_spans
from pipeline.deduplicate import deduplicate


def analyze(spans: list[LineSpan]) -> list[ErrorSpan]:
    ml_errors = _run_ml(spans)
    citation_errors = check_citations(spans)
    entity_errors = check_entities(spans)

    merged = merge_spans(ml_errors, citation_errors, entity_errors)
    deduped = deduplicate(merged)

    return _sort_reading_order(deduped)


def _run_ml(spans: list[LineSpan]) -> list[ErrorSpan]:
    chunks = build_chunks(spans)
    label_id_sequences = predict(chunks)
    return build_error_spans(chunks, label_id_sequences, spans)


def _sort_reading_order(errors: list[ErrorSpan]) -> list[ErrorSpan]:
    """page by page, top-to-bottom, left-to-right - the order a human reading the document would hit them."""
    return sorted(errors, key=lambda e: (e.page_no, e.y0, e.x0))