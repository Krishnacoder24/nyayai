# architecture

## what nyayai does

takes a PDF of an Indian legal document (FIR, contract, court notice) and
returns the same PDF with colored highlights over detected errors:
- 🟡 spelling mistakes (`#FFD700`)
- 🟠 grammar errors (`#FFA500`)
- 🔴 wrong IPC/BNS citations (`#FF4444`)
- 🔵 entity inconsistencies — name/place mismatches across the document (`#00BFFF`)

(colors from `config/constants.py`'s `ERROR_COLORS` — the palette above
matches the real code, not an earlier draft.)

---

## system overview

```
user uploads PDF
       │
       ▼
  FastAPI POST /upload
       │  saves file, enqueues Celery task, returns job_id immediately
       ▼
  Celery worker (filesystem broker, SQLite result backend — no Redis)
       │
       ├──► ocr.pipeline.extract(pdf) -> list[LineSpan]
       │         │
       │    ┌────┴────┐
       │    │         │
       │    ▼         ▼
       │  model/   rules/
       │  (ML)     (citation lookup, entity fuzzy-match)
       │    │         │
       │    └────┬────┘
       │         │
       │    pipeline.engine.analyze(spans) -> list[ErrorSpan]
       │    (merge -> dedupe -> reading-order sort)
       │         │
       │    renderer/
       │    (annotate PDF, JSON report, HTML report)
       │         │
       │    services/storage.py
       │    (save all outputs, flat filenames keyed by job_id)
       │
       ▼
  GET /status/{job_id}  (Celery's own state, polled by frontend every 300ms)
       │
       ▼
  GET /result/{job_id}
       │
       ▼
  React frontend (PDF.js canvas + highlight overlay + margin annotation rail)
```

The whole flow is orchestrated by `services/analysis.py`'s
`AnalysisService`, called from `workers/tasks.py`'s `process_pdf` Celery
task. `api/routes/*.py` never call `ocr/`, `model/`, `rules/`, `pipeline/`,
or `renderer/` directly — only `services/`.

---

## packages and responsibilities

### `ocr/`
entry point: `extract(pdf_path) -> list[LineSpan]`

decides per page whether the page has a native text layer or is scanned.
native pages go through pdfplumber. scanned pages go through surya OCR
(GPU). both produce `LineSpan` objects — one per line of text with real
bounding box coordinates. nothing downstream knows or cares which
extractor produced a given span. **status: done.**

runs on: GPU (surya pages), CPU (native pages)
when: first step, blocking, before anything else

### `model/`
entry point: `predict(chunks) -> list[list[int]]`

pure ML inference. takes `LineSpan`s, groups them into ~510-token chunks
via `preprocess.py`, runs `law-ai/InLegalBERT` token classification,
returns BIO label IDs per token. `postprocess.py` reconstructs
`ErrorSpan`s with bboxes from the label sequences.

no database. no regex. no API calls. just tensors in, labels out.
**status: scaffold complete, but there is no fine-tuned checkpoint yet** —
`predict.py` detects the absence of `model/checkpoint/config.json` and
returns all-`O` labels (no errors) rather than crashing. Every call to
`predict()` currently also reloads the model + tokenizer from scratch —
there is no caching across jobs yet (tracked as a known gap).

`model/pipeline.py` also exists in the repo as a full duplicate of what
`pipeline/engine.py` does — it predates the current `pipeline/` package
and is dead code; nothing calls it. See "known gaps" below.

runs on: GPU
when: after OCR, in parallel with rules/

### `rules/`
entry points: `check_citations(spans)`, `check_entities(spans)`

deterministic checkers. no model loading. citation checker uses regex to
extract citation patterns then queries Qdrant via `corpus.search.lookup_section()`
for exact section lookup. entity checker uses spaCy NER + rapidfuzz fuzzy
matching (threshold=85) to find name inconsistencies across the full
document. **status: both done.**

each checker is independent — citation checker doesn't know entity
checker exists. `pipeline/engine.py` calls both directly (hardcoded, not
through a registry yet — see "known gaps").

`rules/cross_reference_checker.py` is currently a 0-byte placeholder for
a planned future checker (catching references like "as mentioned in
paragraph 3" where paragraph 3 doesn't exist).

runs on: CPU
when: after OCR, can run in parallel with model/

### `pipeline/`
entry point: `analyze(spans) -> list[ErrorSpan]`

orchestration only. calls `model.predict` and the two rule checkers,
passes results to `merger.py` and `deduplicate.py`, returns a clean sorted
list of ErrorSpans. **status: done**, though the checker calls are
hardcoded in `engine.py` rather than going through a pluggable registry —
adding a third checker currently means editing `engine.py` directly.

data flow:
```
list[LineSpan]
      │
      ├──► model.predict -> ErrorSpans (ML)
      ├──► rules.citation_checker.check_citations -> ErrorSpans
      ├──► rules.entity_checker.check_entities -> ErrorSpans
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

one-time (or periodic) pipeline that reads raw legal-act PDFs from
`corpus/sources/`, parses them into sections (`corpus/parser.py`), chunks
each section by structural marker — Explanation/Illustration/Exception,
not fixed token windows (`corpus/chunker.py`) — embeds each passage
(`corpus/embeddings.py`, hardcoded to InLegalBERT, not a configurable
choice), and uploads to Qdrant with metadata (`corpus/uploader.py`).

`search.py` provides the query interface (`lookup_section()`) used by
`rules/citation_checker.py`.

**status: infrastructure done, and all six acts now have their own
parser** (`corpus/parsers/{ipc,bns,bnss,cpc,crpc,constitution}.py`), each
registered in `corpus/parser.py`'s `_PARSERS` dict — per #26, deliberately
independent implementations, no shared base class. A clean end-to-end
`scripts/ingest_corpus.py --all` run against a live Qdrant instance is
still the thing to re-verify before calling this fully closed (see the
QA checklist, issue #52). Full detail in `docs/corpus.md`.

### `renderer/`
entry point: `annotate_pdf(pdf_path, errors) -> annotated_pdf_bytes`

takes the original PDF and a list of ErrorSpans, draws colored highlight
boxes at the correct bbox coordinates on each page, returns the annotated
PDF as bytes. also generates a JSON report (`report.py`) and an HTML
report (`html_report.py`). **status: done.** `html_report.py`'s
`_error_row()` used to crash on every report with at least one error (an
invalid f-string format spec) - fixed in #33, with regression tests added.

### `services/`
business logic between routes and packages. `analysis.py`'s
`AnalysisService` orchestrates the full pipeline for one document:
OCR → analyze → render → save. routes call services, services call
packages. routes never call packages directly. **status: done** -
`analysis.py` and `storage.py` are the two files here; report building
lives in `renderer/report.py` and upload validation lives directly in
`api/routes/upload.py`, so there was never a need for separate
`services/report.py` / `services/upload.py` files - the empty
placeholders that used to sit here (never imported anywhere) were removed.

### `api/`
FastAPI app. thin routes that validate input, hand off to services, and
return responses. async job pattern: `POST /upload` returns a `job_id`
immediately, `GET /status/{job_id}` polls (returning Celery's own state
directly, not a custom state machine), `GET /result/{job_id}` returns the
final output once done. **status: done, no authentication.** Full request/
response detail in `docs/api.md`.

### `workers/`
Celery, using the **filesystem transport as broker + SQLite as result
backend** — no Redis, and `docker-compose.yml` no longer defines a Redis
service. `tasks.py`'s `process_pdf` task calls `services.analysis.AnalysisService`. **Both the API process and the
worker process must resolve `BASE_DIR` to the same absolute path** — if
they're launched from different working directories with a relative path
anywhere in the settings, tasks silently queue forever with no error. The
worker must also be started with `-Q pdf_processing` explicitly, or it
won't consume the queue at all.

---

## data flow in detail

### LineSpan
produced by `ocr/`, consumed by `model/` and `rules/`.

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

### ErrorSpan
produced by `model/` and `rules/`, consumed by `pipeline/`, `renderer/`, `api/`.

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
    suggestion: str     # suggested correction (empty until a correction model exists)
    confidence: float   # 0.0 - 1.0
    # highlight_color is derived, not stored: see ERROR_COLORS in config/constants.py
```

There is currently no field on `ErrorSpan` indicating which subsystem
produced it (model vs. citation rule vs. entity rule), and no
`explanation` field for the planned rich-tooltip feature — both are
tracked as open work, not yet implemented.

### BIO label scheme
used internally by `model/` for token classification. See `docs/model.md`
for the full breakdown — this is the one doc in the repo that stayed
accurate to the real implementation throughout development.

---

## coordinate system

all bboxes use pdfplumber's coordinate system:
- origin at top-left of page
- x increases rightward
- y increases downward
- units are PDF points (1 point = 1/72 inch)

surya OCR produces image-pixel coordinates at scale=2.0 (144 DPI). these
are **not** the same as PDF point coordinates. `ocr/router.py` keeps
native and surya pages processed separately so the two coordinate systems
never get mixed within a page.

**reportlab, used by `renderer/annotate_pdf.py` to draw the highlights, uses
the opposite convention** — PDF-native bottom-left origin, y-up.
`annotate_pdf.py` flips y against page height before drawing for exactly
this reason. Get this backwards and every highlight lands on the wrong
half of the page. `HighlightOverlay.jsx` on the frontend does **not** need
this flip, since PDF.js (like pdfplumber) is already top-left, y-down.

---

## async job lifecycle (as actually implemented)

```
POST /upload
  -> validates file (content-type == application/pdf, size <= 50MB, non-empty)
  -> saves to data/uploads/{job_id}.pdf
  -> enqueues celery task: workers.tasks.process_pdf.delay(job_id)
  -> returns {job_id}

[celery worker picks up task, must be running with -Q pdf_processing]
  -> services.analysis.AnalysisService runs:
       ocr.pipeline.extract(pdf) -> spans
       pipeline.engine.analyze(spans) -> errors
       renderer.annotate_pdf.annotate_pdf(pdf, errors) -> annotated bytes
       renderer.report.build_report(...) -> report dict
       renderer.html_report.render_html(...) -> html string
       services.storage saves all outputs under data/outputs/{job_id}_*
  -> Celery's own result backend (SQLite) tracks job state, not a custom field

GET /status/{job_id}
  -> returns Celery's real state: PENDING | STARTED | SUCCESS | FAILURE | RETRY | REVOKED

GET /result/{job_id}
  -> returns { job_id, status, report, annotated_pdf_url, report_html_url, error }
  -> 409 if the job hasn't reached a terminal state yet
```

Full endpoint-by-endpoint detail, including exact response shapes and
error cases, is in `docs/api.md` — this section only covers the shape of
the pipeline, not the wire format.

---

## hardware assumptions

- NVIDIA GPU with ≥ 6GB VRAM (dev: RTX 4050 laptop)
- CUDA 13.2 (torch pinned to `2.4.1+cu124`)
- surya OCR batch size: `RECOGNITION_BATCH_SIZE=32`, `DETECTOR_BATCH_SIZE=4`
- InLegalBERT inference batch size: 8 chunks per forward pass
- Qdrant running locally on port 6333 (via `docker-compose up -d qdrant`)
- **No Redis required.** Celery's filesystem broker + SQLite result
  backend are both plain local files under `data/celery/` — this is a
  deliberate choice to avoid an extra service for a single-machine local
  tool. `docker-compose.yml` has no Redis service.

---

## what does NOT run on GPU

- pdfplumber (native extraction) — CPU
- rules/ (all checkers) — CPU
- pipeline/ (orchestration) — CPU
- renderer/ (PDF annotation) — CPU
- api/ and workers/ — CPU

only surya OCR and InLegalBERT inference touch the GPU.

---

## known limitations & gaps

**not yet built:**
- fine-tuned model weights — `model/checkpoint/` doesn't have real weights
  yet, so ML error detection (spelling/grammar/citation-via-model) returns
  nothing today. Citation and entity checking work independently of this,
  since they're pure rule-based checkers. (tracked: issue #36)
- `ErrorSpan` provenance (`source`) and rich-tooltip `explanation` fields
  (tracked: issues #38, #40)
- entity checker uses `en_core_web_sm`, which handles Indian names poorly
  — a fine-tuned Indian legal NER model would improve this.
- no correction suggestions — `ErrorSpan.suggestion` is empty for
  ML-detected errors; only citation errors get a suggestion (from the
  corpus's `replaced_by` metadata). (tracked: issue #41)
- no authentication on the API — fine for local use, must be added before
  any deployment.
- no real automated test suite yet — most test files are stubs.
  (tracked: issue #50)

**architectural refinements, not blockers** (folded in from `reviews/*.md`
per issue #48 — these were real recommendations from an earlier
architecture review, kept here now that the review docs themselves have
been retired):
- ML inference (`build_chunks` → `predict` → `build_error_spans`) is
  called directly from `pipeline/engine.py`'s `_run_ml()` rather than
  behind a single `model.analyze(spans)` entry point. Fine while it's
  three function calls; if inference grows more complex (multiple
  checkpoints, batching strategy, ONNX), moving it behind one interface
  would keep `engine.py` implementation-agnostic.
- Chunking structural markers (`Explanation`/`Illustration`/`Exception`)
  are hardcoded in `corpus/chunker.py` rather than defined per-parser.
  Fine while every Act shares this structure; would need to move into
  parser-specific configuration if an Act's structure diverges.
- Most magnitude/similarity thresholds already live in
  `config/constants.py`, but `pipeline/deduplicate.py`'s
  `OVERLAP_THRESHOLD` and `rules/entity_checker.py`'s
  `SIMILARITY_THRESHOLD` are still defined locally — minor inconsistency,
  not a functional issue.

The authoritative, up-to-date list of everything above (with GitHub issue
numbers, milestones, and priority) lives in the repo's issue tracker, not
in this file — treat this "known limitations" section as a snapshot, and
the tracker as the source of truth for current status.