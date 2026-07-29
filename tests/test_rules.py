"""
tests for rules/citation_checker.py and rules/entity_checker.py.

citation checking is tested against a mocked corpus (see
tests/conftest.py's mock_lookup_section) so this never needs a live
Qdrant instance or ingested corpus - it's asserting our own regex +
validity logic, not the corpus's contents.

entity checking is tested with a fake spaCy NER (mock_entity_nlp) for the
same reason: spaCy's small English model is genuinely unreliable on
Indian names (see entity_checker.py's own docstring), so asserting
against its real output would test spaCy, not our clustering logic.
"""

from ocr.tokens import LineSpan
from rules.citation_checker import check_citations
from rules.entity_checker import check_entities


# ---------------------------------------------------------------------------
# check_citations
# ---------------------------------------------------------------------------
def _span(text: str, page_no: int = 0) -> LineSpan:
    return LineSpan(
        text=text, page_no=page_no, source="native",
        x0=10, y0=10, x1=200, y1=25,
    )


def test_check_citations_flags_known_repealed_ipc_section(mock_lookup_section):
    # IPC Section 13 is individually omitted (title starts with
    # "[Repealed]") - not just act-wide "repealed" status, which
    # citation_checker deliberately ignores for IPC/CrPC (see its
    # docstring). this is the case that SHOULD be flagged.
    spans = [_span("The accused was charged under Section 13 IPC.")]

    errors = check_citations(spans)

    assert len(errors) == 1
    assert errors[0].error_type == "citation"
    assert "13" in errors[0].text
    assert "repealed" in errors[0].explanation.lower() or "omitted" in errors[0].explanation.lower() \
        or "[repealed]" in errors[0].explanation.lower()


def test_check_citations_ignores_act_wide_repealed_status_for_ipc(mock_lookup_section):
    # IPC Section 302 has status="repealed" in the mock corpus (every IPC
    # section does, since BNS superseded the whole act on 2024-07-01) but
    # its title is a normal title, not an "[Omitted"/"[Repealed" stub.
    # a real FIR citing this for a pre-2024 offence is not an error.
    spans = [_span("Under Section 302 IPC, the accused faces trial.")]

    errors = check_citations(spans)

    assert errors == []


def test_check_citations_passes_known_active_bns_section(mock_lookup_section):
    spans = [_span("Under Section 103 BNS, the accused faces trial.")]

    errors = check_citations(spans)

    assert errors == []


def test_check_citations_flags_genuinely_repealed_bns_section(mock_lookup_section):
    # unlike IPC/CrPC, BNS's own status field IS a real per-section signal
    # (see _lookup_section's docstring) - a repealed BNS section should
    # still be flagged.
    spans = [_span("Under Section 99 BNS, the accused faces trial.")]

    errors = check_citations(spans)

    assert len(errors) == 1
    assert "replaced by section 101" in errors[0].explanation.lower()


def test_check_citations_flags_unknown_section_number(mock_lookup_section):
    spans = [_span("Under Section 9999 IPC, the accused faces trial.")]

    errors = check_citations(spans)

    assert len(errors) == 1
    assert "no section numbered 9999" in errors[0].explanation.lower()


def test_check_citations_skips_gracefully_when_qdrant_unreachable(monkeypatch):
    import rules.citation_checker as citation_checker

    class _UnreachableClient:
        def get_collections(self):
            raise ConnectionError("qdrant is down")

    monkeypatch.setattr(citation_checker, "QdrantClient", lambda url=None: _UnreachableClient())

    spans = [_span("Under Section 302 IPC, the accused faces trial.")]

    # must not raise - citation_checker.py's whole design point is that a
    # down Qdrant degrades to "skip", never crashes the pipeline.
    errors = check_citations(spans)

    assert errors == []


# ---------------------------------------------------------------------------
# check_entities
# ---------------------------------------------------------------------------
def test_check_entities_catches_misspelled_name_pair(mock_entity_nlp):
    span_a = _span("Complainant Ramesh Kumar filed the report.", page_no=0)
    span_b = _span("Witness statement recorded from Rakesh Kumar.", page_no=1)
    spans = [span_a, span_b]

    mock_entity_nlp({
        span_a.text: [("Ramesh Kumar", "PERSON")],
        span_b.text: [("Rakesh Kumar", "PERSON")],  # deliberate misspelling
    })

    errors = check_entities(spans)

    assert len(errors) == 1
    error = errors[0]
    assert error.error_type == "entity"
    # both forms appear exactly once (a tie) - _get_canonical breaks ties
    # by longer string, and both are equal length, so it keeps whichever
    # was encountered first (span_a's "Ramesh Kumar"). the point of this
    # test is that the OTHER one gets flagged as deviating from it.
    assert error.text == "Rakesh Kumar"
    assert error.suggestion == 'should be "Ramesh Kumar"'


def test_check_entities_prefers_more_frequent_form_as_canonical(mock_entity_nlp):
    # "Ramesh Kumar" appears twice, "Rakesh Kumar" once - the single
    # odd-one-out mention should be the one flagged, not the majority form.
    span_a = _span("Complainant Ramesh Kumar filed the report.", page_no=0)
    span_b = _span("Ramesh Kumar again confirmed his statement.", page_no=1)
    span_c = _span("Witness statement recorded from Rakesh Kumar.", page_no=2)
    spans = [span_a, span_b, span_c]

    mock_entity_nlp({
        span_a.text: [("Ramesh Kumar", "PERSON")],
        span_b.text: [("Ramesh Kumar", "PERSON")],
        span_c.text: [("Rakesh Kumar", "PERSON")],
    })

    errors = check_entities(spans)

    assert len(errors) == 1
    assert errors[0].text == "Rakesh Kumar"
    assert errors[0].suggestion == 'should be "Ramesh Kumar"'


def test_check_entities_no_errors_when_names_are_consistent(mock_entity_nlp):
    span_a = _span("Complainant Ramesh Kumar filed the report.", page_no=0)
    span_b = _span("Ramesh Kumar confirmed his statement.", page_no=1)
    spans = [span_a, span_b]

    mock_entity_nlp({
        span_a.text: [("Ramesh Kumar", "PERSON")],
        span_b.text: [("Ramesh Kumar", "PERSON")],
    })

    errors = check_entities(spans)

    assert errors == []


def test_check_entities_never_clusters_across_entity_types(mock_entity_nlp):
    # "Patna" (place) and a similarly-shaped person name should never be
    # compared to each other - only same-type mentions are clustered.
    span_a = _span("The incident occurred in Patna.", page_no=0)
    span_b = _span("The witness Ratna described the scene.", page_no=1)
    spans = [span_a, span_b]

    mock_entity_nlp({
        span_a.text: [("Patna", "GPE")],
        span_b.text: [("Ratna", "PERSON")],
    })

    errors = check_entities(spans)

    assert errors == []


def test_check_entities_returns_empty_when_ner_unavailable(monkeypatch):
    import rules.entity_checker as entity_checker

    def _raise():
        raise OSError("model not found")

    monkeypatch.setattr(entity_checker, "_load_nlp", _raise)

    errors = check_entities([_span("Some FIR text.")])

    assert errors == []