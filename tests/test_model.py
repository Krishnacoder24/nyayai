"""
tests for model/preprocess.py (build_chunks) and model/predict.py
(predict), and their round-trip through model/postprocess.py
(build_error_spans).

everything here runs against a fake tokenizer and a fake model - never
downloads InLegalBERT, never touches a GPU. that's the whole point of
Issue #50's requirement for this file: the round-trip logic (token <->
span bookkeeping, chunk windowing, BIO decoding) is what's under test,
not InLegalBERT's actual predictions - those need real fine-tuned
weights and a fixture GPU/CI story of their own, out of scope here.
"""

import torch

import model.preprocess as preprocess
import model.predict as predict_module
from model.preprocess import build_chunks, Chunk
from model.predict import predict
from model.postprocess import build_error_spans
from model.schemas import LABEL2ID
from ocr.tokens import LineSpan


# ---------------------------------------------------------------------------
# fake tokenizer - one token per word, deterministic ids
# ---------------------------------------------------------------------------
class _FakeEncoding:
    def __init__(self, input_ids, word_ids):
        self.input_ids = input_ids
        self._word_ids = word_ids

    def word_ids(self):
        return self._word_ids


class FakeTokenizer:
    """
    maps each whitespace-split word to exactly one token id, so the
    token<->word (and therefore token<->span) relationship in a test is
    trivial to reason about by hand. a real BERT tokenizer can split one
    word into several subword tokens - preprocess.py's word_id-based
    "first subword only" bookkeeping is what handles that in production;
    this fixture just doesn't need to exercise that specific case to
    verify the chunk-windowing and span-mapping logic.
    """

    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0

    def __init__(self):
        self._next_id = 1000
        self.word_to_id: dict[str, int] = {}

    def _id_for(self, word: str) -> int:
        # same surface word always maps to the same id, distinct words
        # always get distinct ids - good enough determinism for tests
        # without needing a real vocabulary.
        if word not in self.word_to_id:
            self.word_to_id[word] = self._next_id
            self._next_id += 1
        return self.word_to_id[word]

    def __call__(self, words, is_split_into_words=True, add_special_tokens=False):
        assert is_split_into_words
        assert not add_special_tokens
        input_ids = [self._id_for(w) for w in words]
        word_ids = list(range(len(words)))
        return _FakeEncoding(input_ids, word_ids)


def _patch_tokenizer(monkeypatch, tokenizer: FakeTokenizer):
    monkeypatch.setattr(preprocess, "_get_tokenizer", lambda: tokenizer)
    preprocess._CACHED_TOKENIZER = None


# ---------------------------------------------------------------------------
# build_chunks
# ---------------------------------------------------------------------------
def test_build_chunks_maps_tokens_back_to_originating_spans(monkeypatch):
    _patch_tokenizer(monkeypatch, FakeTokenizer())

    spans = [
        LineSpan(text="Sectoin 302 IPC", page_no=0, source="native", x0=0, y0=0, x1=10, y1=10),
        LineSpan(text="is wrong here", page_no=0, source="native", x0=0, y0=10, x1=10, y1=20),
    ]

    chunks = build_chunks(spans)

    assert len(chunks) == 1
    chunk = chunks[0]

    # [CLS] + 6 words + [SEP] = 8 positions
    assert chunk.input_ids[0] == FakeTokenizer.cls_token_id
    assert chunk.input_ids[-1] == FakeTokenizer.sep_token_id
    assert len(chunk.input_ids) == 8
    assert chunk.token_to_span[0] is None    # [CLS]
    assert chunk.token_to_span[-1] is None   # [SEP]
    # first 3 words came from span 0, next 3 from span 1
    assert chunk.token_to_span[1:4] == [0, 0, 0]
    assert chunk.token_to_span[4:7] == [1, 1, 1]
    assert set(chunk.span_indices) == {0, 1}


def test_build_chunks_skips_spans_with_no_words(monkeypatch):
    _patch_tokenizer(monkeypatch, FakeTokenizer())

    spans = [
        LineSpan(text="   ", page_no=0, source="native", x0=0, y0=0, x1=10, y1=10),
        LineSpan(text="real text here", page_no=0, source="native", x0=0, y0=10, x1=10, y1=20),
    ]

    chunks = build_chunks(spans)

    assert len(chunks) == 1
    # only span index 1 should ever appear - the blank span contributed
    # no tokens at all, so it can never show up in span_indices
    assert chunks[0].span_indices == [1]


def test_build_chunks_splits_into_overlapping_windows_for_long_documents(monkeypatch):
    _patch_tokenizer(monkeypatch, FakeTokenizer())
    # force a tiny window so a handful of spans has to split into
    # multiple chunks, without needing hundreds of real words in the test
    monkeypatch.setattr(preprocess, "MAX_TOKENS", 6)   # window = 4
    monkeypatch.setattr(preprocess, "CHUNK_STRIDE", 1)

    spans = [
        LineSpan(text=f"word{i}", page_no=0, source="native", x0=0, y0=i * 10, x1=10, y1=i * 10 + 10)
        for i in range(10)
    ]

    chunks = build_chunks(spans)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.input_ids) <= 6
        assert chunk.input_ids[0] == FakeTokenizer.cls_token_id
        assert chunk.input_ids[-1] == FakeTokenizer.sep_token_id

    # every span must be covered by at least one chunk - the sliding
    # window must never silently drop content at the tail end
    covered = set()
    for chunk in chunks:
        covered.update(chunk.span_indices)
    assert covered == set(range(10))


# ---------------------------------------------------------------------------
# predict - fake model, real BIO decoding logic
# ---------------------------------------------------------------------------
class FakeModelOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    """
    a stand-in for AutoModelForTokenClassification. instead of running
    real InLegalBERT weights, returns hand-built logits that put all
    probability mass on whatever label id the test wants for each
    position - so predict()'s batching/padding/trimming logic is
    exercised for real, while the "prediction" itself is fully
    controlled and deterministic.
    """

    def __init__(self, label_ids_by_chunk: list[list[int]]):
        self._label_ids_by_chunk = label_ids_by_chunk
        self._call_index = 0
        self.num_labels = len(LABEL2ID)

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        logits = torch.full((batch_size, seq_len, self.num_labels), -10.0)
        for row in range(batch_size):
            label_ids = self._label_ids_by_chunk[self._call_index]
            self._call_index += 1
            for pos in range(seq_len):
                label_id = label_ids[pos] if pos < len(label_ids) else LABEL2ID["O"]
                logits[row, pos, label_id] = 10.0
        return FakeModelOutput(logits)


def _patch_model(monkeypatch, label_ids_by_chunk):
    fake_model = FakeModel(label_ids_by_chunk)
    fake_tokenizer = FakeTokenizer()

    monkeypatch.setattr(
        predict_module, "_load_model_and_tokenizer",
        lambda: (fake_model, fake_tokenizer, torch.device("cpu")),
    )


def test_predict_returns_argmax_labels_for_every_token(monkeypatch):
    chunk = Chunk(
        input_ids=[101, 1000, 1001, 1002, 102],
        attention_mask=[1, 1, 1, 1, 1],
        token_to_span=[None, 0, 0, 0, None],
        span_indices=[0],
    )
    expected_labels = [
        LABEL2ID["O"], LABEL2ID["O"], LABEL2ID["B-CITE"], LABEL2ID["O"], LABEL2ID["O"],
    ]
    _patch_model(monkeypatch, [expected_labels])

    result = predict([chunk])

    assert result == [expected_labels]


def test_predict_returns_empty_for_no_chunks():
    assert predict([]) == []


# ---------------------------------------------------------------------------
# full round-trip: build_chunks -> predict -> build_error_spans
# ---------------------------------------------------------------------------
def test_build_chunks_predict_postprocess_round_trip(monkeypatch):
    """
    synthetic end-to-end check: a document with one deliberately-flagged
    word should come out the other end as exactly one ErrorSpan, with the
    right error_type, text, and bbox pulled from the originating LineSpan.
    """
    _patch_tokenizer(monkeypatch, FakeTokenizer())

    spans = [
        LineSpan(text="The accused cited", page_no=0, source="native", x0=0, y0=0, x1=50, y1=10),
        LineSpan(text="Sectoin 302 IPC", page_no=0, source="native", x0=0, y0=10, x1=50, y1=20),
        LineSpan(text="in his statement", page_no=0, source="native", x0=0, y0=20, x1=50, y1=30),
    ]

    chunks = build_chunks(spans)
    assert len(chunks) == 1  # small enough to fit in one chunk at default MAX_TOKENS
    chunk = chunks[0]

    # flag the tokens mapped to span 1 (the "Sectoin 302 IPC" line) as one
    # contiguous B-CITE/I-CITE span, everything else "O" - standard BIO:
    # only the first token of a span gets B-, the rest get I-.
    label_ids = []
    prev_flagged = False
    for span_idx in chunk.token_to_span:
        if span_idx == 1:
            label_ids.append(LABEL2ID["I-CITE"] if prev_flagged else LABEL2ID["B-CITE"])
            prev_flagged = True
        else:
            label_ids.append(LABEL2ID["O"])
            prev_flagged = False

    _patch_model(monkeypatch, [label_ids])

    predicted = predict(chunks)
    errors = build_error_spans(chunks, predicted, spans)

    assert len(errors) == 1
    error = errors[0]
    assert error.error_type == "citation"
    assert error.text == "Sectoin 302 IPC"
    assert error.page_no == 0
    assert error.bbox == (0, 10, 50, 20)