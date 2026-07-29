# NyayAI

An AI-powered error detection tool for Indian legal documents (FIRs, contracts,
court notices). NyayAI ingests a PDF, detects spelling, grammar, and semantic
errors — including wrong IPC/BNS section citations and entity inconsistencies
across a document — and returns an annotated PDF with color-coded highlights
plus a structured report.

**Everything runs locally.** No external OCR or LLM APIs — built for courts,
law firms, and legal aid organisations where document confidentiality and
per-document cost both matter.

**The full pipeline is verified working end to end**: upload a real PDF in the
real frontend → it hits the real FastAPI backend → Celery runs OCR → the
fine-tuned InLegalBERT model + rules (citation + entity + spelling +
cross-reference) → merge/dedupe/sort → renders an annotated PDF and a
report → the frontend polls and displays the real result. This has been
run against an actual sample FIR PDF, start to finish, not just exercised
piece-by-piece in isolation.

---

## Status

| Component | Status |
|---|---|
| OCR (`ocr/`) | ✅ done |
| Model (`model/`) | ✅ done — fine-tuned InLegalBERT checkpoint trained and in place, ML-based error detection is live |
| Corpus (`corpus/`) | ✅ done — all six act parsers (IPC, BNS, BNSS, CPC, CrPC, Constitution) parse the real PDFs; verified IPC→BNS and CrPC→BNSS mapping tables in `corpus/data/` |
| Rules (`rules/`) | ✅ done — citation, entity, spelling, cross-reference checkers, pluggable registry — see known limitations below |
| Pipeline (`pipeline/`) | ✅ done — full PDF-in to annotated-PDF/report-out flow verified end-to-end on a real document |
| Renderer (`renderer/`) | ✅ done — the crashing HTML-report bug is fixed |
| API + workers (`api/`, `workers/`, `services/`) | ✅ done — no auth yet |
| Frontend (`frontend/`) | ✅ done — wired to the real API, not mock data |
| Tests (`tests/`) | ✅ done — real automated suite, see "running the test suite" below |
| Fine-tuning (`train/`) | ✅ done — training run completed, `model/checkpoint/` has real weights |
| Deployment | ⬜ not started |


the model handles spelling/grammar/citation-shape; the rule-based checkers in
`rules/` (citation, entity, spelling, cross-reference - registered in
`rules/registry.py`) handle things that need either an external source of
truth (citation_checker) or whole-document memory the model doesn't have
(entity_checker) since it only ever sees 512 tokens at a time.

---

## Architecture

this is the actual frozen structure — no more reshuffling planned.

```
NyayAI/
├── ocr/                  done - extract(pdf_path) -> list[LineSpan]
│   ├── tokens.py         LineSpan dataclass (one per line, real measured bbox)
│   ├── native_extractor.py   pdfplumber, for pdfs with a text layer
│   ├── surya_extractor.py    surya-ocr, for scanned pages
│   ├── router.py         decides which extractor each page needs
│   └── pipeline.py       ties it together into one extract() call
│
├── model/                done - fine-tuned checkpoint trained and in place
│   ├── schemas.py        ErrorSpan + BIO label scheme
│   ├── preprocess.py     LineSpans -> token chunks (512 tokens, sliding window)
│   ├── predict.py        InLegalBERT inference - loads model/checkpoint/
│   └── postprocess.py    BIO labels -> ErrorSpans with real bboxes
│
├── rules/                done
│   ├── citation_checker.py       regex + corpus.search lookup
│   ├── entity_checker.py         spacy NER + rapidfuzz clustering
│   ├── spelling_checker.py       rule-based legal-vocabulary spell checker
│   ├── cross_reference_checker.py   flags dangling "see paragraph N" references
│   └── registry.py               pluggable list all four checkers are run through
│
├── corpus/                done
│   ├── schemas.py, chunker.py, embeddings.py, uploader.py, search.py
│   ├── parser.py          dispatches to the right act-specific parser
│   ├── parsers/           one file per act - IPC, BNS, BNSS, CPC, CrPC, Constitution
│   └── data/              verified IPC→BNS / CrPC→BNSS mapping tables + schedules
│
├── pipeline/             done - merge -> deduplicate -> reading-order sort
├── renderer/              done - annotated PDF, colors, JSON + HTML report
│
├── services/              done
│   ├── storage.py         job-id based file layout
│   └── analysis.py        orchestrates extract -> analyze -> render -> save
│
├── workers/                done - Celery, no Redis (see below)
├── api/                    done - FastAPI, upload/status/result/health
│
├── frontend/                ✅ wired to the real API (upload/poll/result)
│   └── src/                PDF.js canvas, colored highlight overlay,
│                           margin annotation rail, error sidebar
│                           (mockData.js kept around, unused - api.js is
│                           what App.jsx actually imports now)
│
├── train/                   done - training run completed, checkpoint in model/checkpoint/
├── config/, data/, scripts/, tests/, docs/
├── docker-compose.yml        qdrant only, no redis service
└── README.md
```

---

## setup

**you need:**
- python 3.10 (pinned - see dependency table)
- NVIDIA GPU with CUDA, 6GB+ VRAM (i have an RTX 4050)
- docker (for qdrant)
- node 20+ (for the frontend)

**install:**

```bash
git clone <repo>
cd NyayAI

uv venv
source .venv/bin/activate
uv sync
```

dependency versions are pinned in `pyproject.toml` for a reason - see the table
below before touching any of them, especially surya/transformers/torch.

**system package:**
```bash
sudo apt install poppler-utils
```

**start qdrant:**
```bash
docker-compose up -d qdrant
```
no redis needed - Celery uses a filesystem broker + sqlite result backend
instead (see "async jobs, no redis" below).

**verify GPU works:**
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
should print `True` and your GPU's name. if it prints `False`, stop here and
fix the CUDA/driver setup before going further - surya and InLegalBERT both
expect a working GPU, and everything downstream (batch sizes, `--pool=solo`
below) is tuned assuming this works.

**ingest the legal corpus** (one-time, or after an act gets amended):
```bash
uv run python scripts/ingest_corpus.py --all
```

---

## running

**backend:**
```bash
uv run uvicorn api.main:app --reload
```

**a worker** (needed for anything to actually get processed):
```bash
uv run celery -A workers.celery_app worker --loglevel=info -Q pdf_processing --pool=solo
```
the `-Q pdf_processing` isn't optional - a worker only consumes queues it's
explicitly told to listen on. leaving it off means uploads just sit there
forever with no error at all (found this out the hard way).

`--pool=solo` isn't optional either, for a different reason: without it,
Celery defaults to the `prefork` pool and forks one child process per CPU
core. Each of those child processes independently imports `model.predict`
and `ocr.surya_extractor` and loads its *own* copy of InLegalBERT and
Surya's detection/recognition models onto the GPU the first time it picks
up a task - the module-level caches in those files only dedupe loads
*within* a process, not across forked siblings. On a 6GB card, two or
three prefork children processing documents at the same time is enough
to blow the VRAM budget on its own, independent of any single document's
size. `--pool=solo` runs everything in one process, one task at a time,
which matches this project's single-GPU, single-machine deployment
target anyway.

**frontend:**
```bash
cd frontend
npm install
npm run dev
```
open `http://localhost:5173` - the viewer works end-to-end against the real
backend now (`frontend/src/api.js`: `POST /upload` → poll `GET /status` →
`GET /result`). `mockData.js` is still in the tree but nothing imports it
anymore; `App.jsx` pulls from `api.js`. set `VITE_API_BASE_URL` in `.env` if
the backend isn't on `http://localhost:8000`.

---

## async jobs, no redis

Celery needs a broker (to queue tasks) and a result backend (to store
outcomes). instead of running redis just for this, it's configured with:

- **broker:** the filesystem transport - a queued task is just a file under
  `data/celery/broker/`
- **result backend:** sqlite, via `db+sqlite:///data/celery/results.sqlite`

both are local files, nothing extra to run. if this ever needs to scale past
one machine, it's a one-line swap to `redis://` - nothing in `workers/` or
`api/` cares which broker is configured.

the one real trap: every path here has to be **absolute**, anchored to a
fixed project-root constant - not a relative path. the API process and the
worker process are launched separately and won't reliably share a working
directory, and a relative path resolves against whatever directory each
process happens to be in. tested this directly: with a relative path, a task
gets written to one physical folder while the worker watches a completely
different one - no error, no crash, it just sits "queued" forever.

---

## training the model

`train/` is done and has actually been run - `model/checkpoint/` now has a
real fine-tuned checkpoint (model weights + tokenizer), and
`model/predict.py` loads it instead of falling back to all-`O` labels.

the flow that was used:

```bash
uv run python scripts/generate_data.py --corpus corpus/sources/ --out data/training
uv run python -m train.train
uv run python -m train.evaluate
```
(`make generate-data`, `make train`, `make evaluate` do the same thing.)

`generate_data.py` produces synthetic training data by deliberately
corrupting real, verified legal text — spelling/grammar/citation corruption
applied in that order, since grammar corruption changes token counts and
would invalidate any index-based labels applied before it. `train.py` then
fine-tunes InLegalBERT with the HuggingFace `Trainer` API, saving both model
weights and the tokenizer into `model/checkpoint/` (so a future retrain from
a different base checkpoint can never end up paired with a stale tokenizer).
Hyperparameters in `train.py` are reasonable BERT-fine-tuning defaults, not
empirically tuned for this task - a good next step if detection quality
needs improving is a sweep, not a rewrite.

after any future retrain, remember to `dvc add model/checkpoint` and push
(see "data & model versioning" below) — otherwise the new checkpoint only
exists on whatever machine trained it.

---

## Dependency versions (frozen)

these are pinned in `pyproject.toml` for a reason - `surya-ocr` and `transformers`
in particular have a real, previously-hit incompatibility (`transformers` newer
than `4.48.0` breaks surya's `SuryaOCRConfig` with `KeyError: 'encoder'`). don't
bump any of these without a specific reason to.

| package | version | why it's pinned |
|---|---|---|
| `torch` | `2.4.0+cu124` | matched to `transformers==4.48.0` and the CUDA 13.2 / RTX 4050 setup this was built against |
| `transformers` | `4.48.0` | newer breaks surya's `SuryaOCRConfig` (see above) |
| `surya-ocr` | `0.9.3` | scanned-page OCR fallback |
| `qdrant-client` | `1.17.1` | corpus vector search |
| `fastapi` | `0.115.0` | API layer |
| `celery` | `5.4.0` | async job queue (filesystem broker + sqlite backend, no redis) |
| `pydantic` / `pydantic-settings` | `2.8.2` / `2.5.2` | settings + schemas |

full list, including unpinned (`>=`) utility deps like `rapidfuzz`, `spacy`, and
`pyspellchecker`, is in `pyproject.toml` - that file is the source of truth, this
table is just the ones worth a second look before touching.

---

## running the test suite

```bash
pytest tests/ --ignore=tests/test_qdrant_live.py -v
```

that excludes `tests/test_qdrant_live.py` on purpose - it's a **live** integration
test against a real, ingested Qdrant instance (`docker-compose up -d qdrant` first,
see "setup" above), not part of the regular automated run.

everything else is fully mocked/synthetic - no GPU, no network, no live services:

- `test_rules.py` mocks the corpus lookup and spaCy's NER (see `conftest.py`'s
  `mock_lookup_section` / `mock_entity_nlp`) - it's testing our own
  citation/entity logic, not the corpus's contents or spaCy's accuracy.
- `test_model.py` mocks the tokenizer and the model itself - no InLegalBERT
  weights get loaded, no GPU needed.
- `test_pipeline.py` and `test_api.py` mock the ML/rules layer and Celery
  respectively, so they only ever exercise their own orchestration logic.
- `test_ocr.py` runs the real native-PDF extraction path against a small
  reportlab-generated sample FIR (see `conftest.py`'s `sample_pdf_path`) - it
  does NOT exercise the surya scanned-page path (needs a real scanned PDF, out
  of scope for an automated fixture). for that, see below.
- `test_parser.py` runs the real parsers against the actual act PDFs in
  `corpus/sources/` - the one file in this suite doing real, not mocked, work.

**`test_parser.py` is slow on purpose** - it's parsing six real, 1-3MB legal
PDFs with pdfplumber (each parsed once per run, cached via `lru_cache`), which
costs ~4-5 minutes total. for a fast loop while iterating on anything else:

```bash
pytest tests/ --ignore=tests/test_qdrant_live.py --ignore=tests/test_parser.py -v
```

**manual OCR smoke test against a real document** (not part of the automated
suite - no assertions, just prints what `extract()` sees, useful for eyeballing
a real scanned FIR that's messier than the synthetic fixture):

```bash
make test-ocr FILE=path/to/real_scanned_fir.pdf
```

**one dependency this needs that isn't pinned above:** `httpx`, for FastAPI's
`TestClient` in `test_api.py`. and if `en_core_web_sm` isn't already pulled down:

```bash
python -m spacy download en_core_web_sm
```

---

## current status

- [x] OCR pipeline (pdfplumber + surya, with process-wide model caching so a
      Celery worker doesn't reload weights per document)
- [x] model (InLegalBERT inference wiring, fine-tuned checkpoint trained and loaded)
- [x] rule-based checkers (citation, entity, spelling, cross-reference)
- [x] pipeline orchestration (merge / dedupe / sort, pluggable rule registry)
- [x] renderer (annotated PDF + JSON/HTML reports - crashing bug fixed)
- [x] FastAPI + Celery async jobs (filesystem + sqlite, no redis)
- [x] React frontend, wired to the real API (not mock data)
- [x] corpus ingestion - all six act parsers done (IPC, BNS, BNSS, CPC, CrPC,
      Constitution), verified IPC→BNS and CrPC→BNSS mapping tables
- [x] drop the redis service from docker-compose.yml (filesystem + sqlite broker in use)
- [x] real automated test suite (Issue #50) - `pytest tests/ --ignore=tests/test_qdrant_live.py`
- [x] full pipeline verified end-to-end on a real sample FIR PDF: upload →
      OCR → model + rules → merge/dedupe/sort → annotated PDF + report →
      frontend displays the real result (issue #52's checklist item)
- [x] config/housekeeping cleanup - `.env.example` fixed, `config/settings.py`
      dead scratch notes removed, `model/pipeline.py` and
      `corpus/parsers/base.py` dead code deleted
- [x] fine-tune InLegalBERT - training run completed, `model/checkpoint/` has real weights
- [ ] auth on the API
- [ ] deployment (M6 - not started)

---

## stuff i learned building this

- **LineSpan, not word-level tokens** - pdfplumber and surya both natively
  give you line-level bboxes. trying to go word-by-word was extra complexity
  for no real benefit.
- **surya-ocr 0.9.3 + transformers 4.48.0 is the only combination that
  works** - anything newer than transformers 4.48 breaks surya's
  `SuryaOCRConfig` with a `KeyError: 'encoder'`. surya 0.20+ needs a whole
  separate vLLM server to run, not worth it for a dev setup.
- **subword continuations need `None`, not the span index** - when aligning
  BERT subword tokens back to source lines, only the *first* subword of each
  word should map to a span index. gave every continuation subword the same
  span index at first, which silently corrupted every multi-subword word's
  span boundaries. easy to miss since it only shows up on longer words.
- **grammar corruption has to run before spelling/citation corruption** in
  synthetic training data - it changes token counts, which would invalidate
  any index-based labels applied earlier.
- **`en_core_web_sm` doesn't just misspell names, it mistags their entity
  TYPE** - tested this directly with real sentences. the same person's name
  got tagged `PERSON` in one sentence and `GPE` (place) in another, which
  sends it to an entirely different clustering bucket in `entity_checker.py`
  - so it never even gets compared against its other spelling. this is worse
  than a simple fuzzy-matching miss.
- **kombu's filesystem transport doesn't auto-create its own directories** -
  neither the broker folders nor sqlite's parent directory get created
  automatically. celery just throws `OperationalError: unable to open
  database file` if they're missing.
- **a real IPC PDF is messier than IndiaCode's clean formatting suggests** -
  a 13-page table of contents where every entry looks almost identical to a
  real section start (just missing the closing dash), footnote reference
  digits stuck directly against bracket-wrapped amended sections
  (`7[5. Certain laws not to be affected...`), at least one section number
  missing its period entirely, and repealed sections that don't appear in
  the body text at all - they just get skipped. a naive "match a number
  then a dash" regex catches almost none of this correctly.
- **reportlab and pdf.js disagree about which corner is the origin** -
  reportlab (used server-side for the annotated PDF) is bottom-left,
  y-increases-up, like real PDF coordinate space. pdf.js (used in the
  browser) is top-left, y-increases-down, matching pdfplumber. get this
  backwards and every highlight silently lands on the wrong half of the
  page - verified this against a real page before trusting either one.
- **a fresh `SuryaExtractor()` per document isn't free just because you
  reuse it across pages within one document** - the constructor used to
  build brand-new `DetectionPredictor()`/`RecognitionPredictor()` instances
  every call, and both push real weights onto CUDA. a Celery worker picks
  up the next scanned document before python's GC/torch's caching allocator
  necessarily hands back the previous instance's VRAM, so this manifested
  as "works for the first few PDFs, OOMs later" rather than an immediate
  crash - the same class of bug `model/predict.py`'s module-level cache
  already solved for InLegalBERT, just not yet applied to surya. fixed with
  the same pattern: process-wide cached predictors, instance-scoped (not
  module-level) page-render cache so page 3 of one document can never leak
  into page 3 of the next.
- **a generated schedule table needs its own coverage check, not just a
  smoke test** - the BNSS First Schedule extraction had two real gaps
  (roughly section-numbers 128-162 and 299-322 missing entirely) that a
  PR review caught by actually diffing covered numbers against the
  expected range, not by spot-checking a few entries. re-running the
  generator against the source PDF closed all but a handful of entries
  (128, 129, 130, 138, 307 as of this writing) - worth a real coverage
  assertion in the corpus test suite eventually, not just a manual PR note.
- **writing real assertions against all six real act PDFs (Issue #50)
  surfaced two small, previously-unnoticed things**: `Section.act` is a
  display-cased name (`"CrPC"`, `"Constitution"`), not the dispatch-key
  string (`"CRPC"`, `"CONSTITUTION"`) used to select the parser - harmless,
  but worth knowing before comparing the two directly. and Constitution
  Articles 28, 203, and 366 come back with an empty `.title` despite
  `status="active"` - their marginal side-note title text appears to have
  merged into `.body` instead, likely a two-column PDF layout artifact
  specific to those three. neither blocks anything today; both are
  candidates for a small follow-up issue against
  `corpus/parsers/constitution.py`.

---

## dependencies and why

| package | why |
|---|---|
| pdfplumber | reads text + real bboxes from PDFs with a text layer |
| surya-ocr | OCR for scanned pages, handles Hindi script too |
| InLegalBERT | BERT model pre-trained on Indian legal text |
| qdrant | vector DB for IPC/BNS/BNSS/Constitution/CPC lookups |
| spacy + rapidfuzz | entity NER + fuzzy name/place consistency checking |
| fastapi | backend API |
| celery | async job processing - filesystem broker + sqlite backend, no redis |
| react + pdf.js | render the PDF in-browser and draw highlights on top |

---

## known issues

- surya is slow (~10s per scanned page on a 4050) - async jobs hide this,
  but it's still slow
- `en_core_web_sm` handles Indian names inconsistently (see above) - needs a
  fine-tuned Indian legal NER model eventually
- no correction suggestions for ML-detected errors yet (citations do have
  suggestions, from the corpus payload)
- BNSS First Schedule extraction is nearly, not fully, complete - 5 entries
  (section-numbers 128, 129, 130, 138, 307) are still missing as of this
  writing, down from ~59 missing across two ranges before the last
  generator run - see "stuff i learned" above
- Constitution Articles 28, 203, and 366 come back with an empty `.title`
  (their marginal side-note title merged into `.body` instead) - a small
  parser gap, not a status/repeal issue, caught while writing `test_parser.py`
- no auth on the API - fine for local single-user use, needs fixing before
  any real deployment. output cleanup exists now (`make cleanup-outputs`,
  see `scripts/cleanup_outputs.py`) but isn't scheduled automatically -
  needs a cron entry or systemd timer to actually run periodically.
- deployment (M6) hasn't started - no Dockerfile, no Vercel config yet

---

See `docs/architecture.md` for the full frozen folder structure.

- [InLegalBERT](https://huggingface.co/law-ai/InLegalBERT)
- [surya OCR](https://github.com/VikParuchuri/surya)
- [IndiaCode](https://indiacode.nic.in) - source for IPC, BNS, BNSS, Constitution, CPC PDFs

---

## data & model versioning (DVC)

`data/` and `model/checkpoint/` are gitignored and tracked with DVC instead
(`data.dvc`, `model/checkpoint.dvc`) - both are too large/binary to live
directly in git. The configured remote (`.dvc/config`) is a local
filesystem path, not a shared/cloud remote, so `dvc pull` on a different
machine needs the remote pointed at wherever you actually keep the DVC
storage first (`dvc remote modify local url <path>`).

**`model/checkpoint/` now has a real trained checkpoint in it** (see
"training the model" above) - make sure it's been `dvc add`ed and pushed
(the exact commands are just below) so `dvc pull` on any other machine
actually fetches something real instead of an empty directory.

**On a fresh clone:**
```bash
git clone <repo>
uv sync
dvc pull
```

**After a new training run** (`make train` writes to `model/checkpoint/`):
```bash
dvc add model/checkpoint
git add model/checkpoint.dvc
git commit -m "Update model checkpoint"
dvc push
git push
```