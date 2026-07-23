# api

## overview

nyayai exposes a REST API built with FastAPI. PDF processing is slow
(OCR + model inference can take tens of seconds) so the API uses an async
job pattern — upload returns immediately with a job ID, the client polls
for status, then fetches the result once done.

base URL (local dev): `http://localhost:8000`

**status:** implemented, no authentication yet — see "known gaps" below.

---

## async job flow

```
1. POST /upload
   client sends a PDF
   server saves it to data/uploads/{job_id}.pdf, enqueues a Celery task
   server returns job_id immediately

2. GET /status/{job_id}
   client polls (frontend polls every 300ms — see frontend/src/api.js)
   server returns Celery's own state directly: PENDING | STARTED | SUCCESS | FAILURE | RETRY | REVOKED

3. GET /result/{job_id}
   client fetches once status == "SUCCESS"
   server returns the report dict + download URLs for the annotated PDF and HTML report
```

There is **no custom job state machine** — `GET /status/{job_id}` calls
`celery.result.AsyncResult(job_id, app=celery_app)` and returns its `.status`
directly. The states you'll see are exactly Celery's five/six built-in states,
not `queued`/`processing`/`done`/`failed`. There is no `stage` or `progress`
field — Celery's result backend doesn't track sub-task progress here, so the
client can only know "not done yet" vs. a terminal state.

---

## endpoints

---

### `POST /upload`

Defined in `api/routes/upload.py`.

**request:**
```
Content-Type: multipart/form-data

file: <pdf binary>
```

**response 200:**
```json
{
  "job_id": "3f7a2b1c-9e2a-4c11-9c2e-1a2b3c4d5e6f"
}
```
(`UploadResponse` — just `job_id`, nothing else.)

**response 400:**
```json
{"detail": "only PDF files are accepted"}
```
or
```json
{"detail": "uploaded file is empty"}
```

**response 413:**
```json
{"detail": "file too large (max 50MB)"}
```

These are FastAPI's default `HTTPException` shape — a single `detail` string,
not a custom `{"error": ..., "message": ...}` envelope.

**validation performed:**
- `file.content_type` must be exactly `application/pdf`
- file size checked against `MAX_UPLOAD_BYTES` (`config/constants.py`, 50MB)
- empty file body rejected

**on success:** `job_id = str(uuid.uuid4())`, file saved via
`services.storage.save_upload(job_id, file_bytes)`, and
`workers.tasks.process_pdf.delay(job_id)` enqueues the Celery task.

---

### `GET /status/{job_id}`

Defined in `api/routes/jobs.py`.

**response 200:**
```json
{
  "job_id": "3f7a2b1c-9e2a-4c11-9c2e-1a2b3c4d5e6f",
  "status": "STARTED"
}
```

`status` is one of Celery's own states (`JobStatusResponse.status` is typed
as `Literal["PENDING", "STARTED", "SUCCESS", "FAILURE", "RETRY", "REVOKED"]`
in `api/schemas/response.py`). There is no separate 404 handling for an
unknown `job_id` — Celery's `AsyncResult` for an ID it has never seen
returns `PENDING` by design, which is indistinguishable from "queued but not
started yet." This is a known ambiguity, not a bug — worth knowing about
before building a client that assumes `PENDING` means "definitely exists."

---

### `GET /result/{job_id}`

Defined in `api/routes/jobs.py`.

**response 200 (job succeeded):**
```json
{
  "job_id": "3f7a2b1c-9e2a-4c11-9c2e-1a2b3c4d5e6f",
  "status": "SUCCESS",
  "report": {
    "source_filename": "fir_sample.pdf",
    "total_errors": 2,
    "errors_by_type": {
      "citation": 1,
      "entity": 1
    },
    "errors": [
      {
        "text": "Section 302 IPC",
        "error_type": "citation",
        "page_no": 1,
        "x0": 54.0, "y0": 301.0, "x1": 210.0, "y1": 318.0,
        "suggestion": "verify Section 302 IPC exists and is active",
        "confidence": 0.95,
        "bbox": [54.0, 301.0, 210.0, 318.0],
        "highlight_color": "#FF4444"
      }
    ]
  },
  "annotated_pdf_url": "/files/3f7a2b1c-..._annotated.pdf",
  "report_html_url": "/files/3f7a2b1c-..._report.html",
  "error": null
}
```

`report` is exactly the dict returned by `renderer/report.py`'s
`build_report()` — there is no `stats` object, no `page_count`, and no
`processing_time_s` field; none of these are tracked anywhere in the
pipeline currently. Colors come from `config/constants.py`'s `ERROR_COLORS`
(`spelling` `#FFD700`, `grammar` `#FFA500`, `citation` `#FF4444`, `entity`
`#00BFFF`).

**response when job is still running (HTTP 409):**
```json
{"detail": "job is not finished yet (status: STARTED)"}
```
Any non-`SUCCESS`, non-`FAILURE` status returns `409 Conflict` with this
message — there's no separate `job_not_ready` error code, just this detail
string.

**response when the job failed (still HTTP 200):**
```json
{
  "job_id": "3f7a2b1c-...",
  "status": "FAILURE",
  "report": null,
  "annotated_pdf_url": null,
  "report_html_url": null,
  "error": "<str(exception) from the Celery task>"
}
```
Note this comes back as a normal `200`, not an error status — the failure
is communicated through the `status`/`error` fields in the body, not the
HTTP status code.

**file URLs:** both `annotated_pdf_url` and `report_html_url` are only
populated if the corresponding file actually exists on disk
(`services/storage.py`'s `annotated_pdf_path`/`report_html_path`), and they
point at `/files/{job_id}_annotated.pdf` / `/files/{job_id}_report.html` —
flat filenames under the generic static mount, not a per-job subdirectory
and not a dedicated download endpoint.

---

### Files: `GET /files/{filename}`

There is **no dedicated download route**. `api/main.py` mounts
`data/outputs/` directly as static files:

```python
app.mount("/files", StaticFiles(directory=settings.outputs_dir), name="files")
```

So `data/outputs/{job_id}_annotated.pdf` is reachable at
`/files/{job_id}_annotated.pdf`, and likewise for `{job_id}_report.html`.
There's no custom `Content-Disposition` header, no auth check on this route,
and no per-job access control — anyone who knows or receives a `job_id` can
fetch its outputs. This is fine for local single-user use and is one of the
things that needs to change before any real deployment (see the repo's
GitHub issue tracker, milestone M6).

---

### `GET /health`

Defined in `api/routes/health.py`.

**response 200 (always 200 — there is no degraded/503 state):**
```json
{
  "status": "ok",
  "qdrant": "reachable"
}
```
or
```json
{
  "status": "ok",
  "qdrant": "unreachable"
}
```

That's the entire response shape. There is **no** `redis` field, no
`model_checkpoint` field, no `gpu_available` field, and no `503` response —
`status` is hardcoded to `"ok"` regardless of Qdrant's reachability; only
the `qdrant` field changes. The check itself just instantiates a
`QdrantClient` and calls `get_collections()` inside a `try/except`.

If `qdrant: "unreachable"`, citation checking silently returns an empty
list rather than failing the pipeline (see `rules/citation_checker.py`) —
this endpoint exists specifically to surface that degradation up front
rather than only discovering it mid-analysis.

---

### `/debug/*` — not implemented

`api/routes/debug.py` currently contains only a module docstring
(`"""Development-only API routes."""`) — no actual `APIRouter`, and it
isn't imported or mounted in `api/main.py`. There is no
`GET /debug/spans/{job_id}` or any other debug endpoint live today. This
is a placeholder for a planned feature, not a working route.

---

## error response shape

FastAPI's default shape is used throughout — a single `detail` field:

```json
{"detail": "human readable description"}
```

There are no custom machine-readable error codes (`invalid_file`,
`file_too_large`, `job_not_found`, etc.) anywhere in the codebase today.
If you're building a client, match on HTTP status code + parse `detail` as
a display string, not as a stable enum.

| situation | HTTP status |
|---|---|
| non-PDF content type | 400 |
| empty file | 400 |
| file over 50MB | 413 |
| `/result` requested before job finished | 409 |

---

## running the API

**start Qdrant first** (the only external service the API needs):
```bash
docker-compose up -d qdrant
```
No Redis is required — Celery uses the filesystem broker + SQLite result
backend (see `config/settings.py` / `workers/celery_app.py`). The current
`docker-compose.yml` still defines a `redis` service left over from an
earlier design; it isn't used by anything and is slated for removal.

**start the API:**
```bash
uv run uvicorn api.main:app --reload
```

**start a Celery worker — the `-Q pdf_processing` flag is mandatory:**
```bash
uv run celery -A workers.celery_app worker --loglevel=info -Q pdf_processing
```
A Celery worker only consumes queues it's explicitly told to listen on.
Omitting `-Q pdf_processing` means uploaded PDFs get enqueued but never
picked up — no error, no crash, the job just sits in `PENDING` forever.

All three (Qdrant, API, worker) need to be running for the full pipeline
to work end to end.

---

## configuration

Settings live in `config/settings.py` as a `pydantic_settings.BaseSettings`
class, loaded from `.env`. The fields that matter for the API/worker layer:

```python
class Settings(BaseSettings):
    root_dir: Path
    checkpoint_dir: Path         # model/checkpoint
    corpus_sources_dir: Path     # corpus/sources
    uploads_dir: Path            # data/uploads
    outputs_dir: Path            # data/outputs
    cache_dir: Path
    temp_dir: Path

    celery_broker_url: str = "filesystem://"
    celery_broker_data_folder: str
    celery_result_backend: str   # db+sqlite:///...

    bert_checkpoint: str = "law-ai/InLegalBERT"
    spacy_model: str = "en_core_web_sm"

    recognition_batch_size: int = 32   # alias: RECOGNITION_BATCH_SIZE
    detector_batch_size: int = 4       # alias: DETECTOR_BATCH_SIZE
    torch_device: str = "cuda"         # alias: TORCH_DEVICE

    qdrant_url: str = "http://localhost:6333"    # alias: QDRANT_URL
    qdrant_collection: str = "legal_corpus"       # alias: QDRANT_COLLECTION
    redis_url: str = "redis://localhost:6379/0"   # alias: REDIS_URL — unused, pending removal

    debug: bool = True   # alias: DEBUG — currently defaults True, worth
                          # flipping to False before any non-local deployment
```

`MAX_UPLOAD_BYTES` (50MB) and `ERROR_COLORS` live separately in
`config/constants.py`, not in `Settings`.

Override any setting via `.env` or environment variables. See `.env.example`
for the variables that matter for local dev (Qdrant URL, surya batch sizes,
torch device).

---

## frontend integration

`frontend/src/api.js` is the real client — it does **not** use relative
`fetch()` calls; it reads an absolute base URL:

```javascript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export async function uploadPdf(file) {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${API_BASE_URL}/upload`, { method: 'POST', body: formData })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || `upload failed (${res.status})`)
  return { jobId: data.job_id }
}

export async function pollJobStatus(jobId) {
  const res = await fetch(`${API_BASE_URL}/status/${jobId}`)
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || `status check failed (${res.status})`)
  return { status: data.status }   // Celery's real state string, e.g. "SUCCESS"
}
```

`App.jsx` polls every **300ms** (`POLL_INTERVAL_MS = 300`) and checks for
the literal strings `'SUCCESS'` and `'FAILURE'` — not `'done'`/`'failed'`.
`fetchResult()` in `api.js` reshapes the response slightly: it spreads
`data.report` at the top level and rewrites `annotated_pdf_url` /
`report_html_url` into absolute URLs against `API_BASE_URL`, so components
downstream (`HighlightOverlay.jsx`, `PdfCanvas.jsx`) don't need to know the
API's base URL themselves.

CORS is currently locked to `http://localhost:5173` (Vite's default dev
port) in `api/main.py` — this will need to change to the real frontend
origin before deployment.

---

## output file lifecycle

```
data/uploads/{job_id}.pdf
data/outputs/{job_id}_annotated.pdf
data/outputs/{job_id}_report.json
data/outputs/{job_id}_report.html
```

Flat filenames keyed by `job_id`, all under `services/storage.py` — no
per-job subdirectory. Both input and output are kept indefinitely; there is
currently **no cleanup task**. `data/uploads/` and `data/outputs/` will grow
unbounded over time. This is a known, tracked gap (see the GitHub issue
tracker) — not implemented yet.

---

## known gaps (not yet implemented)

- **No authentication.** Every route is fully public — fine for local
  single-user use, not fine once this is reachable over a network.
- **No output cleanup task.** Uploads and outputs accumulate forever.
- **No rate limiting.**
- **`/debug/*` routes are stubbed, not real.**
- **Timing middleware (`api/middleware/timing.py`) is a stub** — no
  `X-Process-Time` header is actually added to responses yet, despite the
  file's docstring suggesting it should be.
