"""
single entry point for phase 2.

takes list[LineSpan] from ocr.pipeline.extract() and returns
list[ErrorSpan] combining ML model predictions, citation checks,
and entity consistency checks.

execution order:
  1. preprocess spans into chunks
  2. ML model inference (spelling + grammar + citation via InLegalBERT)
  3. citation checker (rule-based, catches what model missed)
  4. entity checker (document-level, full pass)
  5. deduplicate overlapping spans
  6. sort by reading order and return
"""

import logging

from ocr.tokens import LineSpan
from model.schemas import ErrorSpan
from model.preprocess import build_chunks
from model.predict import predict
from model.postprocess import build_error_spans
from rules.citation_checker import check_citations
from rules.entity_checker import check_entities

logger = logging.getLogger(__name__)


def analyze(spans: list[LineSpan]) -> list[ErrorSpan]:
    """
    spans  - list[LineSpan] from ocr.pipeline.extract()
    returns list[ErrorSpan] sorted by (page_no, y0, x0)
    """
    if not spans:
        return []

    errors: list[ErrorSpan] = []

    # --- ML model pass ---
    # preprocess -> predict -> postprocess
    # returns all O labels (no errors) until fine-tuned weights
    # are dropped into model/checkpoint/
    chunks = build_chunks(spans)
    label_id_sequences = predict(chunks)
    ml_errors = build_error_spans(chunks, label_id_sequences, spans)
    errors.extend(ml_errors)
    logger.info(f"ML model found {len(ml_errors)} errors")

    # --- citation checker ---
    # rule-based, works right now without any fine-tuned weights
    # skips spans already flagged as CITE by the ML model to avoid duplicates
    ml_cited_spans = {id(e) for e in ml_errors if e.error_type == "citation"}
    citation_errors = check_citations(spans)
    errors.extend(citation_errors)
    logger.info(f"citation checker found {len(citation_errors)} errors")

    # --- entity checker ---
    # document-level pass, independent of ML model
    entity_errors = check_entities(spans)
    errors.extend(entity_errors)
    logger.info(f"entity checker found {len(entity_errors)} errors")

    # --- deduplicate ---
    errors = _deduplicate(errors)

    # --- sort by reading order ---
    errors.sort(key=lambda e: (e.page_no, e.y0, e.x0))

    logger.info(f"total errors after dedup: {len(errors)}")
    return errors


def _deduplicate(errors: list[ErrorSpan]) -> list[ErrorSpan]:
    """
    removes overlapping spans keeping the one with higher confidence.
    two spans overlap if they're on the same page and their bboxes intersect.
    this handles the case where ML model and citation checker both flag
    the same citation span.
    """
    if not errors:
        return []

    # sort by confidence descending so we always keep the higher confidence one
    sorted_errors = sorted(errors, key=lambda e: e.confidence, reverse=True)
    kept: list[ErrorSpan] = []

    for candidate in sorted_errors:
        overlaps = any(_bboxes_overlap(candidate, existing) for existing in kept)
        if not overlaps:
            kept.append(candidate)

    return kept


def _bboxes_overlap(a: ErrorSpan, b: ErrorSpan) -> bool:
    if a.page_no != b.page_no:
        return False
    # standard 2D rectangle overlap check
    return not (a.x1 <= b.x0 or b.x1 <= a.x0 or a.y1 <= b.y0 or b.y1 <= a.y0)