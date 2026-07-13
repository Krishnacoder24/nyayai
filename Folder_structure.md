NyayAI/
├── README.md
├── PHASE_1.md
├── PHASE_2.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .env
├── .env.example
├── .gitignore
├── Makefile
├── docker-compose.yml          # qdrant + redis only
│
├── config/
│   ├── __init__.py
│   ├── settings.py             # pydantic BaseSettings, all env vars in one place
│   ├── log_config.py           # logging setup (NOT logging.py - shadows stdlib)
│   └── constants.py            # CHUNK_SIZE, STRIDE, SIMILARITY_THRESHOLD etc
│
├── ocr/                        # phase 1 - complete
│   ├── __init__.py
│   ├── tokens.py               # LineSpan dataclass
│   ├── native_extractor.py
│   ├── surya_extractor.py
│   ├── router.py
│   └── pipeline.py             # extract(pdf_path) -> list[LineSpan]
│
├── model/                      # ML inference only - no regex, no db, no API
│   ├── __init__.py
│   ├── schemas.py              # ErrorSpan + LABELS/LABEL2ID/ID2LABEL together
│   ├── preprocess.py           # LineSpans -> Chunks
│   ├── predict.py              # InLegalBERT inference
│   ├── postprocess.py          # BIO labels -> ErrorSpans
│   └── checkpoint/             # fine-tuned weights (gitignored)
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── special_tokens_map.json
│       ├── vocab.txt
│       └── model.safetensors
│
├── rules/                      # rule-based checkers - no ML, no model loading
│   ├── __init__.py
│   ├── citation_checker.py     # regex + qdrant exact lookup
│   ├── entity_checker.py       # NER + rapidfuzz consistency
│   ├── date_checker.py         # date format + logical consistency (later)
│   ├── formatting_checker.py   # para numbering, section headers (later)
│   ├── abbreviation_checker.py # IPC/BNS used before definition (later)
│   ├── cross_reference_checker.py  # exhibit/annexure references (later)
│   └── consistency_checker.py  # clause-level contradictions (later)
│
├── corpus/                     # IPC/BNS/Constitution ingestion pipeline
│   ├── __init__.py
│   ├── ingest.py               # top level: parse -> chunk -> embed -> upload
│   ├── parser.py               # extracts sections from raw PDF/text
│   ├── chunker.py              # splits sections into passages
│   ├── embeddings.py           # wraps embedding model
│   ├── uploader.py             # pushes to qdrant with metadata
│   ├── search.py               # search helpers used by citation_checker
│   ├── schemas.py              # corpus-specific dataclasses (Section, Passage)
│   └── sources/                # raw legal text files (gitignored, large)
│       ├── ipc/
│       ├── bns/
│       ├── bnss/               # replaces CrPC - NOT the same as BNS
│       ├── constitution/
│       └── cpc/
│
├── pipeline/                   # orchestration only - no business logic here
│   ├── __init__.py
│   ├── engine.py               # analyze(spans) -> list[ErrorSpan]
│   ├── merger.py               # combines ML + rule errors
│   └── deduplicate.py          # removes overlapping spans by confidence
│
├── renderer/                   # output generation from ErrorSpan objects
│   ├── __init__.py
│   ├── annotate_pdf.py         # draws highlight boxes on original PDF
│   ├── colors.py               # error_type -> highlight color
│   ├── report.py               # structured JSON report
│   └── html_report.py          # standalone HTML report
│
├── train/                      # fine-tuning only, never runs at inference
│   ├── __init__.py
│   ├── dataset.py              # loads train/val/test jsonl
│   ├── collator.py             # DataCollatorForTokenClassification
│   ├── train.py                # HuggingFace Trainer setup
│   ├── metrics.py              # seqeval span-level F1
│   └── evaluate.py             # runs eval on test set, prints classification report
│
├── services/                   # business logic layer between routes and packages
│   ├── __init__.py
│   ├── analysis.py             # orchestrates: OCR -> pipeline -> render -> save
│   ├── storage.py              # file save/load, upload/output path management
│   ├── report.py               # report generation service
│   └── upload.py               # upload validation, virus scan hook etc
│
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app
│   ├── dependencies.py         # shared deps (settings, db connections)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── upload.py           # POST /upload -> services/upload
│   │   ├── jobs.py             # GET /status/{job_id}, GET /result/{job_id}
│   │   ├── health.py           # GET /health
│   │   └── debug.py            # dev-only endpoints
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── upload.py           # UploadRequest, UploadResponse
│   │   ├── response.py         # JobResult, ErrorSpanResponse
│   │   └── errors.py           # HTTPError shapes
│   └── middleware/
│       ├── __init__.py
│       └── timing.py           # adds X-Process-Time header
│
├── workers/
│   ├── __init__.py
│   ├── celery_app.py           # celery + redis config
│   ├── tasks.py                # process_pdf task -> services/analysis
│   └── queues.py               # queue names + routing keys
│
├── utils/                      # genuinely reusable helpers only
│   ├── __init__.py             # if this grows beyond ~5 files something is wrong
│   ├── bbox.py                 # bbox overlap, merge, area helpers
│   ├── text.py                 # text normalization, cleaning helpers
│   └── pdf.py                  # pdf page count, metadata helpers
│
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── api.js              # upload / poll / fetch result
│       ├── PdfCanvas.jsx       # renders PDF pages via pdf.js
│       ├── HighlightOverlay.jsx
│       └── UploadPage.jsx
│
├── data/
│   ├── uploads/                # gitignored
│   ├── outputs/                # gitignored
│   ├── training/               # train.jsonl, val.jsonl, test.jsonl - gitignored
│   ├── cache/                  # model cache - gitignored
│   └── temp/                   # scratch - gitignored
│
├── assets/
│   ├── logo/
│   ├── screenshots/
│   └── samples/                # demo PDFs for README/docs
│
├── tests/
│   ├── conftest.py             # shared fixtures
│   ├── test_ocr.py
│   ├── test_model.py
│   ├── test_rules.py
│   ├── test_pipeline.py
│   └── test_api.py
│
├── scripts/
│   ├── download_models.py      # pulls InLegalBERT + spacy model on first setup
│   ├── ingest_corpus.py        # thin wrapper: corpus/ingest.py
│   ├── benchmark.py            # speed + accuracy benchmarks
│   └── smoke_test.py           # end-to-end: pdf in -> errors out, no server needed
│
└── docs/
    ├── architecture.md
    ├── model.md
    ├── corpus.md
    ├── api.md
    └── roadmap.md