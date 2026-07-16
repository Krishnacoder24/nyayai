"""
removes overlapping ErrorSpans, keeping the higher-confidence one.

the main source of duplicates: preprocess.py's sliding window
(build_chunks) overlaps chunks by CHUNK_STRIDE tokens, so a LineSpan near
a chunk boundary can get run through the model twice - once as the tail
of one chunk, once as the head of the next - and flagged twice.

only spans of the SAME error_type on the SAME page are ever compared -
a spelling error and a citation error can legitimately share the same
location (e.g. a misspelled section number is both), so overlap across
different error types is never treated as a duplicate.
"""

from model.schemas import ErrorSpan
from utils.bbox import iou

OVERLAP_THRESHOLD = 0.5  # IoU at or above this = "same error, detected twice"


def deduplicate(spans: list[ErrorSpan]) -> list[ErrorSpan]:
    groups: dict[tuple[int, str], list[ErrorSpan]] = {}
    for span in spans:
        key = (span.page_no, span.error_type)
        groups.setdefault(key, []).append(span)

    result = []
    for group in groups.values():
        result.extend(_dedupe_group(group))
    return result


def _dedupe_group(group: list[ErrorSpan]) -> list[ErrorSpan]:
    """
    greedy: highest confidence first. a span is kept unless it overlaps
    (IoU >= threshold) with a span already kept - which, since we're
    going in descending confidence order, is always a higher-confidence
    span than the one being considered.
    """
    ordered = sorted(group, key=lambda s: s.confidence, reverse=True)
    kept: list[ErrorSpan] = []

    for span in ordered:
        if not any(iou(span.bbox, k.bbox) >= OVERLAP_THRESHOLD for k in kept):
            kept.append(span)

    return kept