NyayAI/
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .env                        # gitignored, not committed
├── .env.example
├── .gitignore
├── .dvc/                       # DVC metadata - local remote configured, see README's "data & model versioning" section
├── .dvcignore
├── data.dvc                    # DVC-tracked pointer to data/
├── Makefile                    # test-ocr, cleanup-outputs, download-models, ingest-corpus, generate-data, train, evaluate targets
├── docker-compose.yml          # qdrant only, no redis service
│
├── config/
│   ├── __init__.py
│   ├── settings.py             # pydantic BaseSettings, all env vars in one place
│   ├── log_config.py           # logging setup (NOT logging.py - shadows stdlib)
│   └── constants.py            # MAX_UPLOAD_BYTES, ERROR_COLORS, MODEL_NAME, BATCH_SIZE, etc.
│
├── ocr/                         # done
│   ├── __init__.py
│   ├── tokens.py                # LineSpan dataclass
│   ├── native_extractor.py
│   ├── surya_extractor.py
│   ├── router.py
│   └── pipeline.py               # extract(pdf_path) -> list[LineSpan]
│
├── model/                       # scaffold done, no fine-tuned weights yet
│   ├── __init__.py
│   ├── schemas.py                # ErrorSpan + LABELS/LABEL2ID/ID2LABEL
│   ├── preprocess.py             # LineSpans -> Chunks
│   ├── predict.py                # InLegalBERT inference; model+tokenizer cached after first call (#35)
│   ├── postprocess.py            # BIO labels -> ErrorSpans
│   ├── .gitignore                 # /checkpoint - actual weights tracked via DVC instead, see checkpoint.dvc
│   ├── checkpoint.dvc             # DVC pointer for model/checkpoint/ - see README's "data & model versioning"
│   └── checkpoint/                # gitignored, populated by `dvc pull` or `make train`
│
├── rules/                        # citation, entity, and cross-reference checkers done, registered in registry.py
│   ├── __init__.py
│   ├── citation_checker.py        # done - regex + qdrant exact lookup via corpus.search
│   ├── entity_checker.py          # done - NER + rapidfuzz consistency
│   ├── cross_reference_checker.py # done - exhibit/annexure reference checking
│   └── registry.py                # RULES list - add a new checker here only, engine.py never changes
│   (date_checker.py, formatting_checker.py, abbreviation_checker.py, consistency_checker.py
│    are planned future checkers with no file yet - not stubbed, just not started)
│
├── corpus/                        # infra done; all six Act parsers built (IPC, BNS, BNSS, CPC, CRPC, Constitution)
│   ├── __init__.py
│   ├── ingest.py                  # top level: parse -> chunk -> embed -> upload
│   │                              #   NOTE: has a stray unused `from surya import settings` import,
│   │                              #   shadowed by the real `from config.settings import settings` - dead import
│   ├── parser.py                  # dispatch only; _PARSERS dict currently only registers IPC
│   ├── pdf_utils.py                # shared PDF text-extraction + header-stripping helpers -
│   │                              #   single source of truth now (issue #26); parsers call these
│   │                              #   instead of keeping private copies
│   ├── chunker.py                  # splits Section.body by legal structure (Explanation/
│   │                              #   Illustration/Exception markers), not by token count
│   ├── embeddings.py               # wraps InLegalBERT (hardcoded, not a configurable choice);
│   │                              #   file's own top comment incorrectly says "legal-bert-base-uncased" - stale
│   ├── uploader.py                 # pushes to qdrant with metadata payload;
│   │                              #   get_client() hardcodes localhost:6333, ignores settings.qdrant_url
│   ├── search.py                   # lookup_section() - the only sanctioned way rules/ touches Qdrant
│   ├── schemas.py                  # Section / Passage dataclasses (fields: act, unit_type, number,
│   │                              #   title, body/text, status, metadata dict)
│   └── parsers/
│       ├── ipc.py                   # TOC-guided rewrite done (issue #25) - handles footnote/bracket
│       │                            #   noise, missing periods, letter-suffixed chapters (VA/IXA/XXA)
│       ├── bns.py                   # 0-byte placeholder
│       ├── bnss.py                  # 0-byte placeholder
│       ├── cpc.py                   # 0-byte placeholder
│       └── constitution.py          # 0-byte placeholder
│   └── sources/                   # raw legal text files (gitignored, large)
│       ├── ipc/
│       ├── bns/
│       ├── bnss/                    # replaces CrPC - NOT the same as BNS
│       ├── constitution/
│       └── cpc/
│
├── pipeline/                      # done
│   ├── __init__.py
│   ├── engine.py                    # analyze(spans) -> list[ErrorSpan]; calls model.predict +
│   │                                #   every rule in rules/registry.py's RULES list
│   ├── merger.py                    # combines ML + rule errors
│   └── deduplicate.py                # removes overlapping spans by confidence
│
├── renderer/                      # done
│   ├── __init__.py
│   ├── annotate_pdf.py               # draws highlight boxes on original PDF
│   ├── colors.py                     # error_type -> highlight color
│   ├── report.py                     # structured JSON report (build_report())
│   └── html_report.py                # HTML report; _error_row() crash fixed (#33)
│
├── train/                         # scaffolded; data/training/*.jsonl has been generated
│                                  #   (scripts/generate_data.py), fine-tuning run itself tracked in issue #36
│   ├── __init__.py
│   ├── dataset.py                    # loads train/val/test jsonl
│   ├── collator.py                   # DataCollatorForTokenClassification
│   ├── train.py                      # HuggingFace Trainer setup
│   ├── metrics.py                    # seqeval span-level F1
│   └── evaluate.py                   # runs eval on test set, prints classification report
│
├── services/                      # analysis.py + storage.py done and in active use;
│                                  #   report.py and upload.py are 0-byte placeholders, not wired
│                                  #   anywhere - upload validation currently lives in api/routes/upload.py
│   ├── __init__.py
│   ├── analysis.py                   # AnalysisService: orchestrates OCR -> pipeline -> render -> save
│   ├── storage.py                    # file save/load; flat filenames keyed by job_id under data/uploads, data/outputs
│   ├── report.py                     # 0-byte placeholder
│   └── upload.py                     # 0-byte placeholder
│
├── api/                           # done, no auth
│   ├── __init__.py
│   ├── main.py                       # FastAPI app; CORS currently hardcoded to localhost:5173;
│   │                                #   mounts data/outputs at /files via StaticFiles
│   ├── dependencies.py               # shared deps (settings, etc.)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── upload.py                 # POST /upload - validates + enqueues Celery task
│   │   ├── jobs.py                   # GET /status/{job_id}, GET /result/{job_id}
│   │   ├── health.py                 # GET /health - checks Qdrant reachability only
│   │   └── debug.py                  # GET /debug/queue, POST /debug/jobs/{job_id}/force-status -
│   │                                #   only mounted when settings.debug is True (see main.py)
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── upload.py                 # UploadResponse (job_id only)
│   │   └── response.py               # JobStatusResponse, JobResultResponse - status is Celery's own state literal
│   └── middleware/
│       ├── __init__.py
│       └── timing.py                  # adds an X-Process-Time header to every response,
│                                     #   wired in via app.add_middleware() in main.py
│
├── workers/                       # done - filesystem broker + SQLite result backend, NOT Redis
│   ├── __init__.py
│   ├── celery_app.py                 # celery config; despite the name, no Redis involved
│   ├── tasks.py                       # process_pdf task -> services.analysis.AnalysisService
│   └── queues.py                      # queue name: pdf_processing - worker MUST be started with -Q pdf_processing
│
├── utils/                          # capped at ~5 files by design
│   ├── __init__.py
│   ├── bbox.py                       # bbox overlap, merge, area helpers
│   ├── text.py                       # text normalization, cleaning helpers
│   └── pdf.py                        # pdf page count, metadata helpers
│
├── frontend/                       # done, fully wired to the real API (not mock data)
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx                     # polls /status every 300ms, checks for 'SUCCESS'/'FAILURE'
│       ├── api.js                      # real fetch calls against VITE_API_BASE_URL
│       ├── PdfCanvas.jsx               # renders PDF pages via pdf.js
│       ├── HighlightOverlay.jsx        # native `title` tooltip today - rich popover is a planned feature
│       └── UploadPage.jsx
│
├── data/
│   ├── uploads/                     # gitignored - cleaned up by scripts/cleanup_outputs.py (OUTPUT_RETENTION_DAYS)
│   ├── outputs/                     # gitignored - same
│   ├── training/                    # train.jsonl, val.jsonl, test.jsonl - gitignored, not yet generated
│   ├── cache/                       # model cache - gitignored
│   ├── temp/                        # scratch - gitignored
│   └── celery/                      # filesystem broker + sqlite result backend files live here
│
│
├── tests/                          # NO REAL TESTS YET - see note below
│   ├── conftest.py                    # empty (`pass`)
│   ├── test_ocr.py                    # manual print-script, no assertions
│   ├── test_parser.py                 # manual print-script, no assertions
│   ├── test_model.py                  # empty (`pass`)
│   ├── test_rules.py                  # empty (`pass`)
│   ├── test_pipeline.py               # empty (`pass`)
│   └── test_api.py                    # empty (`pass`)
│
├── scripts/
│   ├── ingest_corpus.py               # thin wrapper: corpus/ingest.py
│   ├── generate_data.py               # synthetic training data corruption - run via `make generate-data`
│   ├── download_models.py             # pre-caches InLegalBERT + en_core_web_sm - run via `make download-models`
│   ├── cleanup_outputs.py             # deletes uploads/outputs older than OUTPUT_RETENTION_DAYS - cron/timer-invoked, --dry-run flag
│   ├── benchmark.py                   # OCR/inference speed + accuracy, reuses train/evaluate.py's scoring path
│   ├── smoke_test.py                  # end-to-end pdf-in/errors-out check, no docker/celery/FastAPI needed
│   ├── test_deps.py                   # confirms pinned dependency versions actually resolved
│   └── test_gpu.py                    # confirms torch can see the GPU
│
└── docs/
    ├── architecture.md
    ├── model.md
    ├── corpus.md
    ├── api.md
    ├── roadmap.md
    └──  Folder_structure.md            # this file

---

## notes on this listing vs. the repo
- none currently — DVC is documented in README.md's "data & model
  versioning" section and in active use with a configured local remote.