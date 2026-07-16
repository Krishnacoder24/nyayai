"""
combines ML-detected errors (postprocess.py) with rule-based errors
(citation_checker.py, entity_checker.py) into a single list.

all three sources already return list[ErrorSpan] - same shape regardless
of origin - so merging is just concatenation. kept as its own step
(rather than inlined in engine.py) so there's a clear seam later if we
ever need to tag provenance or weight one source over another.
"""

from model.schemas import ErrorSpan


def merge_spans(*span_lists: list[ErrorSpan]) -> list[ErrorSpan]:
    merged = []
    for spans in span_lists:
        merged.extend(spans)
    return merged