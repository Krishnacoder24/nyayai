> **Status: historical dev log, meaningfully superseded in places.** This
> records the original phase-2 plan and reasoning. Two things drifted from
> what's described here: (1) `citation_checker.py` and `entity_checker.py`
> ended up living in `rules/`, not `model/`, once that package was split
> out; (2) the single entry point described as `model/pipeline.py` was
> superseded by `pipeline/engine.py` — `model/pipeline.py` still exists in
> the repo today as a leftover duplicate and is dead code, tracked for
> removal. The BIO label scheme, preprocessing, and inference design
> described below are still accurate to the real `model/` code — see
> `docs/model.md` for the current, verified version of that content.

# phase 2 — model / error detection (`feature/model`)

## goal

take the `list[LineSpan]` that phase 1 produces and return a
`list[ErrorSpan]` — each span carrying the flagged text, error type,
bounding box, and confidence score.

single public API:
```python
from model.pipeline import analyze

errors = analyze(spans)  # spans is list[LineSpan] from ocr.pipeline.extract()
# returns list[ErrorSpan]
```

---

## honest picture before start

**InLegalBERT has no error-detection head out of the box.**
`law-ai/InLegalBERT` is pretrained on Indian legal text (good — it
understands legal vocabulary, citations, IPC references). but its
existing task heads are for judgment outcome prediction and document
segmentation — not "is this word misspelled" or "is this section number
wrong." that head doesn't exist yet.

so phase 2 builds the full inference scaffold with the correct
architecture. before fine-tuned weights exist, `predict.py` returns all
`O` (no error) labels — honest, not fake. the moment you drop fine-tuned
weights into `model/checkpoint/`, real predictions start flowing through
without changing a single line of code elsewhere.

fine-tuning itself (generating synthetic data + training) is a separate
branch (`feature/finetune`) that runs in parallel and feeds weights back
into this scaffold.

**exception: citation checking and entity checking don't need fine-tuning.**
`citation_checker.py` is pure retrieval — regex extracts citation spans,
Qdrant checks them against the IPC/BNS/Constitution corpus.
`entity_checker.py` is pure NER + fuzzy matching — no training needed,
works on real documents right now.

---

## files to build (in order)

```
model/
├── __init__.py
├── schemas.py           DONE - ErrorSpan dataclass, BIO label definitions
├── preprocess.py        groups LineSpans into model-ready chunks
├── predict.py           loads InLegalBERT, runs token classification
├── postprocess.py       merges BIO token labels into ErrorSpans
├── citation_checker.py  checks citation spans against Qdrant
├── entity_checker.py    checks entity consistency across the full document
└── pipeline.py          analyze(spans) -> list[ErrorSpan]
```

---

## what each file does

### `schemas.py` — done
defines `ErrorSpan` and the BIO label scheme used across all files:
- `O` — correct
- `B-SPELL / I-SPELL` — spelling error
- `B-GRAM / I-GRAM` — grammar error
- `B-CITE / I-CITE` — wrong IPC/BNS citation
- `B-ENT / I-ENT` — entity inconsistency (petitioner name mismatch,
  wrong place name, conflicting evidence reference)

`LABEL2ID` and `ID2LABEL` live here as the single source of truth.
every other file imports from here — no label mapping defined anywhere else.

---

### `preprocess.py`
**what it does:**
InLegalBERT is a BERT model with a 512 subword token limit. a long legal
document has hundreds of lines. preprocess groups `LineSpan`s into chunks
that fit within that limit, and critically keeps a mapping from each
subword token back to its source `LineSpan` — so when predict.py says
"token 47 is B-CITE", postprocess can look up which LineSpan that token
came from and get its bbox.

**inputs:** `list[LineSpan]`
**outputs:** list of `Chunk` objects, each containing:
- tokenized input ids, attention mask (ready to pass to the model)
- a `token_to_span` mapping: `list[int | None]` where index is token
  position and value is the index into the original spans list
  (`None` for [CLS], [SEP], and subword continuations)

**industry practice — subword alignment:**
BERT doesn't tokenize by whitespace. "Section" might become ["Section"]
but "IPC" might become ["IP", "##C"]. only the first subword of each
word gets a real label — continuations get `-100` so they're ignored
in loss computation during training. this is the standard HuggingFace
convention and `word_ids()` from the tokenizer is how you implement it.

**industry practice — stride/overlap for long documents:**
rather than hard-cutting at 512 tokens (losing context at boundaries),
use a sliding window with overlap — e.g. chunks of 512 tokens with 128
token stride. this way a citation that spans a chunk boundary still gets
seen in full context by at least one chunk.

---

### `predict.py`
**what it does:**
loads `law-ai/InLegalBERT` with a token classification head
(`AutoModelForTokenClassification`), runs inference on preprocessed
chunks, returns raw label IDs per token.

**inputs:** list of `Chunk` objects from preprocess
**outputs:** list of label ID sequences, one per chunk

**before fine-tuned weights exist:**
looks for weights at `model/checkpoint/`. if not found, logs a warning
and returns all `O` labels (no errors detected). this is the correct
honest behavior — not an exception, not fake predictions.

**after fine-tuned weights exist:**
drop weights into `model/checkpoint/` and predictions become real.
nothing else changes.

**industry practice — inference only:**
always call `model.eval()` and wrap inference in `torch.no_grad()`.
skips gradient computation entirely, roughly halves memory usage, speeds
up inference. not optional, not "nice to have" — forgetting this is a
common bug that silently wastes vram.

**industry practice — device handling:**
check `torch.cuda.is_available()` and move model + inputs to GPU if
available, CPU otherwise. never hardcode `"cuda"` — makes the code
break on machines without a GPU.

**industry practice — batching:**
don't run one chunk at a time through the model. batch multiple chunks
per forward pass up to the vram limit. for the 4050 6gb card, batch
size 8 is a reasonable default for inference (larger than training
since we don't need to store gradients).

---

### `postprocess.py`
**what it does:**
takes raw per-token label IDs from predict.py and the chunk objects from
preprocess.py and reconstructs actual `ErrorSpan`s with real bboxes.

steps:
1. for each token with a `B-` label, start a new span
2. extend it with following `I-` tokens of the same type
3. collect the source `LineSpan` bboxes for all tokens in the span
4. merge bboxes into one tight box covering the full error span
5. construct `ErrorSpan` with merged bbox, error type, confidence

**inputs:** chunks + raw label sequences from predict
**outputs:** `list[ErrorSpan]`

**industry practice — confidence from logits:**
don't just take argmax of logits for the label. also run softmax and
store the probability of the predicted class as `confidence` on the
`ErrorSpan`. gives the frontend something to threshold on — e.g. only
show errors with confidence > 0.7.

**industry practice — bbox merging:**
when a span covers multiple tokens that might come from different lines
(rare but possible at chunk boundaries), take:
- `x0 = min(all x0s)`
- `x1 = max(all x1s)`
- `y0 = min(all y0s)`
- `y1 = max(all y1s)`

this gives a tight bounding box that covers all tokens in the span.

---

### `citation_checker.py`
**what it does:**
independently of the ML model, extracts citation patterns from spans
using regex and checks them against the IPC/BNS/Constitution corpus
stored in Qdrant. flags citations that don't match any known valid
section as `CITE` errors.

patterns it looks for:
- `Section \d+ IPC`
- `Section \d+ BNS`
- `u/s \d+`
- `Article \d+ of the Constitution`

for each match, queries Qdrant for that section number. if no match or
section is repealed post-BNS, flags it as a citation error.

**inputs:** `list[LineSpan]`
**outputs:** `list[ErrorSpan]` (citation errors only)

**industry practice — don't duplicate ML model's work:**
citation checker only runs on spans the ML model didn't already flag as
`CITE` errors. in `pipeline.py`, ML errors come first, citation checker
fills in any citations the model missed. no double-flagging.

**industry practice — Qdrant connection handling:**
wrap Qdrant client calls in try/except. if Qdrant is down or the
collection doesn't exist yet (corpus not ingested), log a warning and
return empty list — don't crash the whole pipeline because the vector
db isn't running.

---

### `entity_checker.py`
**what it does:**
a document-level consistency checker, completely independent of the ML
token classifier. the token classifier is local — it sees 512 tokens at
a time and has no memory of what appeared on page 1 when it's processing
page 3. entity consistency is a document-level problem that needs a
separate approach.

**the problem it solves:**
an FIR where the petitioner is "Ramesh Kumar" on page 1 but "Rakesh Kumar"
on page 3 is a serious legal error. similarly: a witness referred to as
"Anwar Sheikh" in the complaint but "Anwar Shaikh" in the evidence list,
or a location written as "Patna" in one place and "Patana" elsewhere.

**how it works:**
1. NER pass — run a NER model over all spans to extract named entities:
   persons (petitioner, respondent, witness, judge), places, organizations,
   evidence references. the Kalamkar corpus has fine-tuned NER models for
   Indian legal text with 14 entity types including PETITIONER, RESPONDENT,
   WITNESS, COURT, STATUTE — use one of these, not a generic English NER.
2. clustering — group mentions that refer to the same real-world entity
   using fuzzy string matching (rapidfuzz edit distance). "Ramesh Kumar"
   and "Rakesh Kumar" have edit distance 1, likely the same person.
3. canonical form — pick the most frequent mention as the canonical name.
4. flag deviations — any mention that differs from canonical form beyond
   a threshold is flagged as an `ENT` error, with the canonical form as
   the suggestion.

**inputs:** `list[LineSpan]` (full document, not chunks)
**outputs:** `list[ErrorSpan]` (entity inconsistency errors only)

**industry practice — NER model choice:**
don't use spacy's default English NER — it doesn't know "IPC", "FIR",
"BNS", Indian names, or Indian place names. use a model fine-tuned on
Indian legal text. options:
- `law-ai/InLegalBERT` fine-tuned on Kalamkar NER corpus
- `ai4bharat/indner` for multilingual Indian NER
the NER model is separate from the error-detection classifier — two
different heads, two different tasks.

**industry practice — fuzzy threshold tuning:**
edit distance threshold for "same entity" needs tuning per entity type.
person names: threshold ~2 (typos, transpositions). place names: threshold
~1 (stricter, "Patna" vs "Patana" is almost certainly a typo but "Patna"
vs "Pune" is a different city). start conservative, tune on real FIRs.

**industry practice — don't flag intentional variations:**
"Supreme Court of India" and "the Court" are the same entity but not
a consistency error — one is a formal reference, one is a pronoun-like
shorthand. filter out short generic references before clustering.
minimum entity mention length of ~3 characters is a reasonable floor.

**industry practice — graceful degradation:**
if the NER model isn't available or fails, return empty list. entity
checking is valuable but not critical path — the pipeline should still
return spelling/grammar/citation errors even if entity checking is down.

---

### `pipeline.py`
**what it does:**
the single entry point for phase 2. calls preprocess → predict →
postprocess for ML errors, then citation checker, then entity checker,
deduplicates overlapping spans, and returns the final sorted list.

```python
from model.pipeline import analyze
errors = analyze(spans)  # list[LineSpan] in, list[ErrorSpan] out
```

**execution order:**
1. ML token classifier (spelling + grammar + citation via model)
2. citation checker (catches citations model missed, cross-checks Qdrant)
3. entity checker (document-level consistency, full document pass)
4. deduplicate overlapping spans (ML + citation checker might both flag same span)
5. sort by (page_no, y0, x0) and return

**industry practice — deduplication:**
ML model and citation checker might flag the same span. before returning,
deduplicate by checking bbox overlap. keep the one with higher confidence.

**industry practice — sort output:**
return errors sorted by `(page_no, y0, x0)` — reading order. makes the
frontend's job trivial and makes debugging output human-readable.

---

## updated label scheme (schemas.py needs one addition)

```python
LABELS = [
    "O",
    "B-SPELL", "I-SPELL",   # spelling errors
    "B-GRAM",  "I-GRAM",    # grammar errors
    "B-CITE",  "I-CITE",    # wrong IPC/BNS citations
    "B-ENT",   "I-ENT",     # entity inconsistency
]
```

`B-ENT / I-ENT` is added to cover entity errors from `entity_checker.py`
so all error types share the same `ErrorSpan` structure and the frontend
only needs to handle one data shape.

---

## done when

```python
from ocr.pipeline import extract
from model.pipeline import analyze

spans = extract("some_fir.pdf")
errors = analyze(spans)

for e in errors:
    print(e.page_no, e.error_type, e.text, e.bbox, e.confidence)
```

runs without crashing. citation and entity errors are real and detectable
right now. spelling and grammar errors return empty until fine-tuned
weights land in `model/checkpoint/`.

---

## what this phase does NOT do

- train the model (that's `feature/finetune`)
- ingest the IPC/BNS corpus into Qdrant (that's `feature/corpus`)
- generate suggested corrections (future work, separate model)
- serve results over HTTP (that's `feature/api`)
