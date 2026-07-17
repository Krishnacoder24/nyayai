# NyayAI

An AI-powered error detection tool for Indian legal documents (FIRs, contracts,
court notices). NyayAI ingests a PDF, detects spelling, grammar, and semantic
errors — including wrong IPC/BNS section citations and entity inconsistencies
across a document — and returns an annotated PDF with color-coded highlights
plus a structured report.

**Everything runs locally.** No external OCR or LLM APIs — built for courts,
law firms, and legal aid organisations where document confidentiality and
per-document cost both matter.

---

## Status

| Component | Status |
|---|---|
| OCR (`ocr/`) | ✅ done |
| Model scaffold (`model/`) | ✅ done — no fine-tuned weights yet, returns all-`O` labels |
| Corpus (`corpus/`) | 🟡 infra done (schemas, chunker, embeddings, uploader, search) — act-specific parsers in progress |
| Rules (`rules/`) | ✅ done — see known limitations below |
| Pipeline (`pipeline/`) | ✅ done |
| Renderer (`renderer/`) | ✅ done |
| API + workers (`api/`, `workers/`, `services/`) | ✅ done |
| Frontend (`frontend/`) | 🟡 scaffolded — running against mock data, not yet wired to the real API |
| Fine-tuning (`train/`) | ⬜ not started |


the model handles spelling/grammar/citation-shape; the two rule-based checkers
handle things that need either an external source of truth (citation_checker)
or whole-document memory the model doesn't have (entity_checker) since it only
ever sees 512 tokens at a time.

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
├── model/                done (scaffold) - no fine-tuned weights yet
│   ├── schemas.py        ErrorSpan + BIO label scheme
│   ├── preprocess.py     LineSpans -> token chunks (512 tokens, sliding window)
│   ├── predict.py        InLegalBERT inference - returns all-O until a checkpoint exists
│   └── postprocess.py    BIO labels -> ErrorSpans with real bboxes
│
├── rules/                done
│   ├── citation_checker.py   regex + corpus.search lookup
│   └── entity_checker.py     spacy NER + rapidfuzz clustering
│
├── corpus/                🟡 infra done, act-specific parsers in progress
│   ├── schemas.py, chunker.py, embeddings.py, uploader.py, search.py
│   └── parsers/          one parser per act (no shared base class - each
│                          act's PDF has different grammar and different
│                          real-world formatting quirks, see below)
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
├── frontend/                🟡 scaffolded, running on mock data
│   └── src/                PDF.js canvas, colored highlight overlay,
│                           margin annotation rail, error sidebar
│
├── train/                   ⬜ not started yet
├── config/, data/, scripts/, tests/, docs/
├── docker-compose.yml        qdrant only (dropping redis - see below)
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
docker-compose up -d qdrant
```

**ingest the legal corpus** (one-time, or after an act gets amended):
```bash
uv run python scripts/ingest_corpus.py --all
```
The `-Q pdf_processing` is required — see the note above.

---

## running

**backend:**
```bash
uv run uvicorn api.main:app --reload
```

**a worker** (needed for anything to actually get processed):
```bash
uv run celery -A workers.celery_app worker --loglevel=info -Q pdf_processing
```
the `-Q pdf_processing` isn't optional - a worker only consumes queues it's
explicitly told to listen on. leaving it off means uploads just sit there
forever with no error at all (found this out the hard way).

**frontend:**
```bash
cd frontend
npm install
npm run dev
```
open `http://localhost:5173` - currently shows the viewer working end-to-end
against mock error data, not real backend results yet.

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

not built yet. `train/` is empty. the plan (per the roadmap) is to generate
synthetic training data by deliberately corrupting real, verified legal text
(spelling/grammar/citation corruption applied in that order, since grammar
corruption changes token counts and would invalidate any index-based labels
applied before it), then fine-tune InLegalBERT with the HuggingFace Trainer
API. no numbers to report yet since none of this has actually run.

---

## Dependency versions (frozen)

```bash
pytest tests/ -v
```

---

## current status

- [x] OCR pipeline (pdfplumber + surya)
- [x] model scaffold (InLegalBERT inference wiring - no fine-tuned weights)
- [x] rule-based checkers (citation + entity consistency)
- [x] pipeline orchestration (merge / dedupe / sort)
- [x] renderer (annotated PDF + JSON/HTML reports)
- [x] FastAPI + Celery async jobs (filesystem + sqlite, no redis)
- [x] React frontend scaffold (PDF.js viewer, mock data)
- [ ] corpus ingestion - infra done, act-specific parsers in progress
- [ ] connect frontend to the real API (currently mock data)
- [ ] drop the now-unused redis service from docker-compose.yml
- [ ] fine-tune InLegalBERT (need to generate training data first)

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
- no fine-tuned weights yet, so spelling/grammar/citation-shape detection via
  the model returns nothing until `train/` exists
- no correction suggestions for ML-detected errors yet (citations do have
  suggestions, from the corpus payload)
- IPC parser still in progress - see "stuff i learned" above for why it's
  taking a while to get right
- frontend is real now but running on mock data, not the actual backend yet
- no auth on the API, no cleanup task for old uploads/outputs - both fine for
  local single-user use, both need fixing before any real deployment

---

See `docs/architecture.md` for the full frozen folder structure.

- [InLegalBERT](https://huggingface.co/law-ai/InLegalBERT)
- [surya OCR](https://github.com/VikParuchuri/surya)
- [IndiaCode](https://indiacode.nic.in) - source for IPC, BNS, BNSS, Constitution, CPC PDFs
