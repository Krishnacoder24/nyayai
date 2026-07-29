"""
tests for pipeline/merger.py, pipeline/deduplicate.py, and the
analyze() orchestration in pipeline/engine.py.

analyze() itself is tested with the ML side and the rule checkers both
mocked out (pipeline.engine._run_ml and pipeline.engine.RULES) - the
thing this file is actually responsible for is merge -> dedupe -> sort,
not re-testing the ML pipeline (tests/test_model.py) or any individual
rule checker (tests/test_rules.py).
"""

from model.schemas import ErrorSpan
from pipeline.merger import merge_spans
from pipeline.deduplicate import deduplicate
import pipeline.engine as engine
from ocr.tokens import LineSpan


def _error(error_type="citation", page_no=0, bbox=(0, 0, 10, 10), confidence=0.5, text="x"):
    x0, y0, x1, y1 = bbox
    return ErrorSpan(
        text=text, error_type=error_type, page_no=page_no,
        x0=x0, y0=y0, x1=x1, y1=y1, confidence=confidence,
    )


# ---------------------------------------------------------------------------
# merge_spans
# ---------------------------------------------------------------------------
def test_merge_spans_concatenates_all_lists_in_order():
    a = [_error(text="a1"), _error(text="a2")]
    b = [_error(text="b1")]
    c: list[ErrorSpan] = []

    merged = merge_spans(a, b, c)

    assert [e.text for e in merged] == ["a1", "a2", "b1"]


def test_merge_spans_with_no_lists_returns_empty():
    assert merge_spans() == []


# ---------------------------------------------------------------------------
# deduplicate
# ---------------------------------------------------------------------------
def test_deduplicate_keeps_higher_confidence_of_two_overlapping_same_type():
    low = _error(error_type="citation", bbox=(0, 0, 10, 10), confidence=0.4, text="low")
    high = _error(error_type="citation", bbox=(1, 1, 11, 11), confidence=0.9, text="high")

    result = deduplicate([low, high])

    assert len(result) == 1
    assert result[0].text == "high"


def test_deduplicate_keeps_both_when_different_error_types():
    # a misspelled section number could legitimately be both a spelling
    # error and a citation error at the exact same location - see
    # pipeline/deduplicate.py's own docstring.
    citation = _error(error_type="citation", bbox=(0, 0, 10, 10), confidence=0.9, text="cite")
    spelling = _error(error_type="spelling", bbox=(0, 0, 10, 10), confidence=0.9, text="spell")

    result = deduplicate([citation, spelling])

    assert len(result) == 2
    assert {e.text for e in result} == {"cite", "spell"}


def test_deduplicate_keeps_both_when_different_pages():
    page1 = _error(page_no=0, bbox=(0, 0, 10, 10), confidence=0.9, text="p1")
    page2 = _error(page_no=1, bbox=(0, 0, 10, 10), confidence=0.9, text="p2")

    result = deduplicate([page1, page2])

    assert len(result) == 2


def test_deduplicate_keeps_both_when_no_real_overlap():
    left = _error(bbox=(0, 0, 10, 10), confidence=0.5, text="left")
    right = _error(bbox=(100, 100, 110, 110), confidence=0.5, text="right")

    result = deduplicate([left, right])

    assert len(result) == 2


# ---------------------------------------------------------------------------
# analyze() - full merge/dedupe/sort orchestration
# ---------------------------------------------------------------------------
def test_analyze_merges_dedupes_and_sorts_in_reading_order(monkeypatch):
    spans = [LineSpan(text="doesn't matter for this test", page_no=0, source="native",
                       x0=0, y0=0, x1=10, y1=10)]

    # ML side "detects" a citation error on page 1, plus a duplicate of
    # a rule-detected spelling error on page 0 (lower confidence, so the
    # rule's higher-confidence version should win after dedup)
    ml_errors = [
        _error(error_type="citation", page_no=1, bbox=(0, 50, 20, 60), confidence=0.6, text="ml-page1"),
        _error(error_type="spelling", page_no=0, bbox=(0, 0, 10, 10), confidence=0.3, text="ml-dup-low-conf"),
    ]
    monkeypatch.setattr(engine, "_run_ml", lambda spans: ml_errors)

    def _rule_top(spans):
        return [_error(error_type="citation", page_no=0, bbox=(0, 0, 10, 10), confidence=0.95, text="rule-top")]

    def _rule_bottom(spans):
        return [
            _error(error_type="spelling", page_no=0, bbox=(0, 0, 10, 10), confidence=0.9, text="rule-dup-high-conf"),
            _error(error_type="citation", page_no=0, bbox=(0, 100, 10, 110), confidence=0.8, text="rule-bottom"),
        ]

    monkeypatch.setattr(engine, "RULES", [_rule_top, _rule_bottom])

    result = engine.analyze(spans)

    texts = [e.text for e in result]

    # the low-confidence ML duplicate must have lost to the higher-
    # confidence rule-detected spelling error at the same location
    assert "ml-dup-low-conf" not in texts
    assert "rule-dup-high-conf" in texts

    # everything else survives (different page/type/location, no overlap)
    assert "ml-page1" in texts
    assert "rule-top" in texts
    assert "rule-bottom" in texts
    assert len(result) == 4

    # reading order: page 0 top-to-bottom before page 1, and within page 0
    # top (y0=0, x0=0 - "rule-dup-high-conf" and "rule-top" tie here and
    # keep their relative merge order) before bottom (y0=100)
    assert texts == ["rule-dup-high-conf", "rule-top", "rule-bottom", "ml-page1"]


def test_analyze_returns_empty_when_nothing_detected(monkeypatch):
    monkeypatch.setattr(engine, "_run_ml", lambda spans: [])
    monkeypatch.setattr(engine, "RULES", [lambda spans: []])

    result = engine.analyze([])

    assert result == []