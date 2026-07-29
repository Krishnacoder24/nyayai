"""
shared fixtures for the whole test suite.

kept deliberately small (see Issue #50) - fixtures that only one test file
needs live in that file instead of bloating this one for everybody.
"""

import io
import os

import pytest

from config.settings import settings
from ocr.tokens import LineSpan


# ---------------------------------------------------------------------------
# filesystem setup
# ---------------------------------------------------------------------------
# api/main.py mounts StaticFiles(directory=settings.outputs_dir) at import
# time, and workers/celery_app.py creates its broker/result-backend
# directories at import time too - both raise/fail if the directories don't
# already exist. production relies on these having been created once
# (e.g. by `make` / first run); tests need the same guarantee before
# anything imports api.main, so this runs for every test session
# regardless of which test file goes first.
@pytest.fixture(scope="session", autouse=True)
def _ensure_data_dirs():
    for path in (
        settings.uploads_dir,
        settings.outputs_dir,
        settings.cache_dir,
        settings.temp_dir,
    ):
        os.makedirs(path, exist_ok=True)
    os.makedirs(settings.celery_broker_data_folder, exist_ok=True)
    os.makedirs(os.path.join(settings.celery_broker_data_folder, "..", "processed"), exist_ok=True)


# ---------------------------------------------------------------------------
# sample PDF
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """
    a small, real, valid single-page PDF containing FIR-like text - built
    with reportlab so it's a genuine PDF (real xref table, real content
    stream), not a hand-rolled string that happens to start with %PDF.
    good enough for ocr.pipeline.extract() and for API upload-flow tests;
    not meant to exercise OCR accuracy (see tests/test_ocr.py for that,
    it needs a real scanned FIR).
    """
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica", 12)
    lines = [
        "FIRST INFORMATION REPORT",
        "Police Station: Kotwali, District: Patna",
        "FIR No: 145/2024",
        "Under Section 302 IPC and Section 103 BNS",
        "Complainant: Ramesh Kumar, S/O Late Suresh Kumar",
        "The complainant Rakesh Kumar stated that on the night of the incident,",
        "he witnessed the accused near the scene as described in paragraph 1.",
    ]
    y = 800
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture
def sample_pdf_path(tmp_path, sample_pdf_bytes):
    path = tmp_path / "sample_fir.pdf"
    path.write_bytes(sample_pdf_bytes)
    return path


# ---------------------------------------------------------------------------
# mock LineSpan list
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_line_spans() -> list[LineSpan]:
    """
    a small hand-built FIR-like document as LineSpans, skipping OCR
    entirely. deliberately contains:
      - two citations ("Section 302 IPC", "Section 103 BNS")
      - an entity inconsistency ("Ramesh Kumar" vs "Rakesh Kumar")
      - a paragraph reference ("paragraph 1") that IS defined, so
        cross-reference checks over this fixture stay quiet by default -
        tests that want a dangling reference build their own small list.
    """
    return [
        LineSpan(
            text="FIRST INFORMATION REPORT",
            page_no=0, source="native",
            x0=72, y0=40, x1=300, y1=55,
            is_heading=True,
        ),
        LineSpan(
            text="1. Under Section 302 IPC and Section 103 BNS",
            page_no=0, source="native",
            x0=72, y0=60, x1=400, y1=75,
        ),
        LineSpan(
            text="Complainant: Ramesh Kumar, S/O Late Suresh Kumar",
            page_no=0, source="native",
            x0=72, y0=80, x1=420, y1=95,
        ),
        LineSpan(
            text="The complainant Rakesh Kumar stated that on the night of the incident",
            page_no=0, source="native",
            x0=72, y0=100, x1=430, y1=115,
        ),
        LineSpan(
            text="he witnessed the accused near the scene, as described in paragraph 1.",
            page_no=0, source="native",
            x0=72, y0=120, x1=430, y1=135,
        ),
    ]


# ---------------------------------------------------------------------------
# mock corpus / Qdrant
# ---------------------------------------------------------------------------
class FakeQdrantClient:
    """
    stand-in for qdrant_client.QdrantClient that never touches the network.
    get_collections() always succeeds (so citation_checker's connectivity
    probe passes); real filter-matching is NOT reimplemented here - tests
    that need lookup_section() behaviour patch that function directly
    (see mock_lookup_section below), since citation_checker never calls
    the client's query methods itself, only corpus.search.lookup_section.
    """

    def get_collections(self):
        return type("Collections", (), {"collections": []})()


@pytest.fixture
def fake_qdrant_client():
    return FakeQdrantClient()


@pytest.fixture
def mock_corpus_sections() -> dict[tuple[str, str], dict]:
    """
    a tiny in-memory (act, number) -> payload table standing in for the
    real Qdrant corpus. mirrors the shape corpus.search.lookup_section
    actually returns (see corpus/schemas.py's Section/Passage fields).
    """
    return {
        ("ipc", "302"): {
            "act": "ipc", "unit_type": "section", "number": "302",
            "title": "Punishment for murder", "status": "repealed",
            "text": "Whoever commits murder shall be punished with death, or imprisonment for life...",
            "metadata": {},
        },
        ("ipc", "13"): {
            # a section individually omitted before IPC itself was ever
            # superseded - see citation_checker._lookup_section's docstring
            # for why this (not `status`) is what makes an IPC citation
            # actually invalid.
            "act": "ipc", "unit_type": "section", "number": "13",
            "title": "[Repealed]", "status": "repealed",
            "text": "", "metadata": {},
        },
        ("bns", "103"): {
            "act": "bns", "unit_type": "section", "number": "103",
            "title": "Punishment for murder", "status": "active",
            "text": "Whoever commits murder shall be punished with death, or imprisonment for life...",
            "metadata": {},
        },
        ("bns", "99"): {
            "act": "bns", "unit_type": "section", "number": "99",
            "title": "Some repealed BNS section", "status": "repealed",
            "text": "",
            "metadata": {"effective_date": "2024-07-01", "replaced_by": "101"},
        },
    }


@pytest.fixture
def mock_lookup_section(monkeypatch, mock_corpus_sections):
    """
    patches rules.citation_checker.lookup_section (and QdrantClient) so
    check_citations() runs against mock_corpus_sections instead of a real
    Qdrant instance. returns the mock_corpus_sections dict for tests that
    want to assert against it directly.
    """
    import rules.citation_checker as citation_checker

    def _fake_lookup_section(number, act, client=None):
        return mock_corpus_sections.get((act.strip().lower(), number))

    monkeypatch.setattr(citation_checker, "lookup_section", _fake_lookup_section)
    monkeypatch.setattr(citation_checker, "QdrantClient", lambda url=None: FakeQdrantClient())
    return mock_corpus_sections


# ---------------------------------------------------------------------------
# fake spaCy NER (for rules/entity_checker.py)
# ---------------------------------------------------------------------------
class _FakeSpan:
    def __init__(self, text, label_):
        self.text = text
        self.label_ = label_


class _FakeDoc:
    def __init__(self, ents):
        self.ents = ents


class FakeNLP:
    """
    stands in for `nlp = spacy.load("en_core_web_sm")`.

    real spaCy's small English model is genuinely unreliable on Indian
    names (entity_checker.py's own docstring admits this) - asserting
    against its actual NER output would make this test flaky and would
    really be testing spaCy's model quality, not our clustering/fuzzy-
    match logic. this fixture makes NER deterministic so the test targets
    what check_entities is actually responsible for: given entity
    mentions, does it correctly cluster and flag the inconsistent one.
    """

    def __init__(self, entities_by_text: dict[str, list[tuple[str, str]]]):
        # entities_by_text: span text -> list of (entity_text, spacy_label) pairs
        self._entities_by_text = entities_by_text

    def __call__(self, text: str) -> _FakeDoc:
        ents = [_FakeSpan(t, label) for t, label in self._entities_by_text.get(text, [])]
        return _FakeDoc(ents)


@pytest.fixture
def mock_entity_nlp(monkeypatch):
    """
    patches rules.entity_checker._load_nlp to return a FakeNLP built from
    whatever mapping the test supplies. usage:

        def test_x(mock_entity_nlp):
            mock_entity_nlp({"line text": [("Ramesh Kumar", "PERSON")]})
            errors = check_entities(spans)
    """
    import rules.entity_checker as entity_checker

    def _apply(entities_by_text: dict[str, list[tuple[str, str]]]):
        fake_nlp = FakeNLP(entities_by_text)
        monkeypatch.setattr(entity_checker, "_load_nlp", lambda: fake_nlp)
        return fake_nlp

    return _apply