# architecture

## what nyayai does

takes a PDF of an Indian legal document (FIR, contract, court notice) and
returns the same PDF with colored highlights over detected errors:
- 🟡 spelling mistakes
- 🟠 grammar errors
- 🔴 wrong IPC/BNS citations
- 🟣 entity inconsistencies (name/place mismatches across the document)

---

## system overview

```
user uploads PDF
       │
       ▼
  FastAPI /upload
       │
       ▼
  Celery task (background)
       │
       ├──► OCR layer
       │         │
       │    list[LineSpan]
       │         │
       │    ┌────┴────┐
       │    │         │
       │    ▼         ▼
       │  model/   rules/
       │  (ML)     (regex + retrieval)
       │    │         │
       │    └────┬────┘
       │         │
       │    list[ErrorSpan]
       │         │
       │    pipeline/engine.py
       │    (merge + dedup)
       │         │
       │    renderer/
       │    (annotate PDF)
       │         │
       │    services/storage.py
       │    (save output)
       │
       ▼
  GET /result/{job_id}
       │
       ▼
  React frontend
  (PDF canvas + highlight overlay)
```

---

## packages and responsibilities

### `ocr/`
entry point: `extract(pdf_path) -> list[LineSpan]`

decides per page whether the page has a native text layer or is scanned.
native pages go through pdfplumber. scanned pages go through surya OCR
(GPU). both produce `LineSpan` objects — one per line of text with real
bounding box coordinates. nothing downstream knows or cares which
extractor produced a given span.

runs on: GPU (surya pages), CPU (native pages)
when: first step, blocking, before anything else

### `model/`
entry point: `predict(chunks) -> list[list[int]]`

pure ML inference. takes `LineSpan`s, groups them into 512-token chunks
via `preprocess.py`, runs `law-ai/InLegalBERT` token classification, 
returns BIO label IDs per token. `postprocess.py` reconstructs
`ErrorSpan`s with bboxes from the label sequences.

no database. no regex. no API calls. just tensors in, labels out.

runs on: GPU
when: after OCR, in parallel with rules/

### `rules/`
entry points: `check_citations(spans)`, `check_entities(spans)`, etc.

deterministic checkers. no model loading. citation checker uses regex to
extract citation patterns then queries Qdrant for exact section lookup.
entity checker uses spacy NER + rapidfuzz fuzzy matching to find name
inconsistencies across the full document.

each checker is independent — citation checker doesn't know entity
checker exists. pipeline/engine.py calls them and merges results.

runs on: CPU
when: after OCR, can run in parallel with model/

### `pipeline/`
entry point: `analyze(spans) -> list[ErrorSpan]`

orchestration only. calls model/ and rules/ checkers, passes results to
merger.py and deduplicate.py, returns a clean sorted list of ErrorSpans.
no business logic here — just coordination.

data flow:
```
list[LineSpan]
      │
      ├──► model/predict -> ErrorSpans (ML)
      ├──► rules/citation_checker -> ErrorSpans
      ├──► rules/entity_checker -> ErrorSpans
      │
      ▼
  merger.py   (combine all error lists)
      │
      ▼
  deduplicate.py  (overlapping spans -> keep highest confidence)
      │
      ▼
  list[ErrorSpan] sorted by (page_no, y0, x0)
```

### `corpus/`
entry point: `ingest.py`

one-time (or periodic) pipeline that reads raw IPC/BNS/Constitution/BNSS
PDFs from `corpus/sources/`, parses them into sections, chunks each
section, embeds via a sentence embedding model, and uploads to Qdrant
with metadata (`section_no`, `act`, `status: active/repealed`).

`search.py` provides the query interface used by `rules/citation_checker`.

### `renderer/`
entry point: `annotate_pdf(pdf_path, errors) -> annotated_pdf_bytes`

takes the original PDF and a list of ErrorSpans, draws colored highlight
boxes at the correct bbox coordinates on each page, returns the annotated
PDF as bytes. also generates a JSON report and optionally an HTML report.

### `services/`
business logic between routes and packages. `analysis.py` orchestrates
the full pipeline for one document: OCR → analyze → render → save.
routes call services, services call packages. routes never call packages
directly.

### `api/`
FastAPI app. thin routes that validate input, hand off to services, and
return responses. async job pattern: POST /upload returns a job_id
immediately, GET /status/{job_id} polls, GET /result/{job_id} returns
the final output once done.

### `workers/`
Celery + Redis. the actual PDF processing runs as a background Celery
task so the API doesn't block on a 30-60 second job. tasks.py calls
services/analysis.py.

---

## data flow in detail

### linespan
produced by ocr/, consumed by model/ and rules/.

```python
@dataclass
class LineSpan:
    text: str        # full line text
    page_no: int
    source: str      # "native" or "surya"
    x0: float        # left edge
    y0: float        # top edge
    x1: float        # right edge
    y1: float        # bottom edge
```

### errorspan
produced by model/ and rules/, consumed by pipeline/, renderer/, api/.

```python
@dataclass
class ErrorSpan:
    text: str           # the flagged text
    error_type: str     # "spelling", "grammar", "citation", "entity"
    page_no: int
    x0: float
    y0: float
    x1: float
    y1: float
    suggestion: str     # suggested correction (empty until correction model added)
    confidence: float   # 0.0 - 1.0
```

### BIO label scheme
used internally by model/ for token classification.

```
O         correct, no error
B-SPELL   beginning of spelling error
I-SPELL   continuation of spelling error
B-GRAM    beginning of grammar error
I-GRAM    continuation
B-CITE    beginning of wrong citation
I-CITE    continuation
B-ENT     beginning of entity inconsistency
I-ENT     continuation
```

---

## coordinate system

all bboxes use pdfplumber's coordinate system:
- origin at top-left of page
- x increases rightward
- y increases downward
- units are PDF points (1 point = 1/72 inch)

surya OCR produces image-pixel coordinates at scale=2.0 (144 DPI). these
are NOT the same as PDF point coordinates. if you ever mix coordinates
from native and surya on the same page, highlights will be misaligned.
this is handled in router.py — native and surya pages are always processed
separately and never mixed within a page.

---

## async job lifecycle

```
POST /upload
  -> validates file
  -> saves to data/uploads/{job_id}.pdf
  -> enqueues celery task
  -> returns {job_id, status: "queued"}

[celery worker picks up task]
  -> services/analysis.py runs:
       extract(pdf) -> spans
       analyze(spans) -> errors
       annotate_pdf(pdf, errors) -> annotated bytes
       save output to data/outputs/{job_id}/
  -> updates job status in redis

GET /status/{job_id}
  -> returns {status: "processing" | "done" | "failed"}

GET /result/{job_id}
  -> returns {
       annotated_pdf_url: "...",
       errors: list[ErrorSpanResponse],
       stats: {total, by_type, processing_time_s}
     }
```

---

## hardware assumptions

- NVIDIA GPU with ≥ 6GB VRAM (dev: RTX 4050 laptop)
- CUDA 12.4+
- surya OCR batch size: RECOGNITION_BATCH_SIZE=32, DETECTOR_BATCH_SIZE=4
- InLegalBERT inference batch size: 8 chunks per forward pass
- Qdrant running locally on port 6333
- Redis running locally on port 6379

---

## what does NOT run on GPU

- pdfplumber (native extraction) — CPU
- rules/ (all checkers) — CPU
- pipeline/ (orchestration) — CPU
- renderer/ (PDF annotation) — CPU
- api/ and workers/ — CPU

only surya OCR and InLegalBERT inference touch the GPU.

---

## known limitations

- surya OCR is slow (~10s per scanned page on RTX 4050)
- InLegalBERT error detection requires fine-tuned weights — base model
  returns no errors until model/checkpoint/ is populated
- entity checker uses en_core_web_sm which handles Indian names poorly —
  needs a fine-tuned Indian legal NER model for production quality
- citation checker needs Qdrant running with corpus ingested —
  returns empty list if Qdrant is unreachable
- no correction suggestions yet — suggestion field is empty until a
  seq2seq correction model is added