NyayAI/
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .env                        # gitignored, not committed
├── .env.example
├── .gitignore
├── .dvc/                       # DVC metadata - present but not documented anywhere else; confirm intended usage
├── .dvcignore
├── data.dvc                    # DVC-tracked pointer to data/ - see note above
├── Makefile                    # currently only has a test-ocr target
├── docker-compose.yml          # still defines a redis service left over from before the filesystem-broker decision - unused, pending removal
├── test_deps.py                # ad hoc root-level script, not in scripts/ or tests/ - dependency-check smoke test
├── test_gpu.py                 # ad hoc root-level script, not in scripts/ or tests/ - GPU/CUDA check
│
├── config/
│   ├── __init__.py
│   ├── settings.py             # pydantic BaseSettings, all env vars in one place
│   │                            #   NOTE: currently has ~30 lines of leftover scratch
│   │                            #   notes + a duplicate BASE_DIR definition appended
│   │                            #   below the real Settings class - needs cleanup
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
│   ├── predict.py                # InLegalBERT inference; reloads model+tokenizer every call, no caching yet
│   ├── postprocess.py            # BIO labels -> ErrorSpans
│   ├── pipeline.py               # DEAD CODE - full duplicate of pipeline/engine.py + merger.py +
│   │                              #   deduplicate.py, predates the pipeline/ package, nothing imports it
│   └── checkpoint/                # empty - fine-tuned weights not yet produced (gitignored once present)
│
├── rules/                        # citation + entity done; several planned checkers are 0-byte placeholders
│   ├── __init__.py
│   ├── citation_checker.py        # done - regex + qdrant exact lookup via corpus.search
│   ├── entity_checker.py          # done - NER + rapidfuzz consistency
│   └── cross_reference_checker.py # 0-byte placeholder - exhibit/annexure reference checking (planned)
│   (date_checker.py, formatting_checker.py, abbreviation_checker.py, consistency_checker.py
│    are planned future checkers with no file yet - not stubbed, just not started)
│
├── corpus/                        # infra done; IPC parser rewritten (issue #25), BNS/BNSS/CPC/Constitution not yet started
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
│   │                                #   rules checkers directly (hardcoded, no registry yet)
│   ├── merger.py                    # combines ML + rule errors
│   └── deduplicate.py                # removes overlapping spans by confidence
│
├── renderer/                      # done, but html_report.py has a live crashing bug
│   ├── __init__.py
│   ├── annotate_pdf.py               # draws highlight boxes on original PDF
│   ├── colors.py                     # error_type -> highlight color
│   ├── report.py                     # structured JSON report (build_report())
│   └── html_report.py                # HTML report; _error_row() has an invalid f-string format
│                                     #   spec and raises ValueError on any report with >= 1 error - P0 bug
│
├── train/                         # scaffolded, never executed
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
│   │   └── debug.py                  # 0-byte placeholder (docstring only) - NOT wired into main.py,
│   │                                #   no debug routes actually exist yet
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── upload.py                 # UploadResponse (job_id only)
│   │   └── response.py               # JobStatusResponse, JobResultResponse - status is Celery's own state literal
│   └── middleware/
│       ├── __init__.py
│       └── timing.py                  # docstring-only stub - NOT added via app.add_middleware(),
│                                     #   no X-Process-Time header is actually added yet
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
│   ├── uploads/                     # gitignored - no cleanup task yet, grows unbounded
│   ├── outputs/                     # gitignored - same
│   ├── training/                    # train.jsonl, val.jsonl, test.jsonl - gitignored, not yet generated
│   ├── cache/                       # model cache - gitignored
│   ├── temp/                        # scratch - gitignored
│   └── celery/                      # filesystem broker + sqlite result backend files live here
│
├── reviews/                        # historical architectural review notes from earlier in
│                                   #   development - some recommendations here are now tracked
│                                   #   as GitHub issues; candidate for folding into architecture.md
│                                   #   and removing once those issues close
│   ├── review of phase 3 - corpus section.md
│   └── review of pipeline and orchestration.md
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
│   └── generate_data.py               # synthetic training data corruption - not yet run
│
└── docs/
    ├── architecture.md
    ├── model.md
    ├── corpus.md
    ├── api.md
    ├── roadmap.md
    ├── Folder_structure.md            # this file
    ├── PHASE_1.md                     # historical dev log - phase 1 (OCR), kept for the real
    │                                 #   lessons learned during that phase, still broadly accurate
    ├── PHASE_2.md                     # historical dev log - phase 2 (model), superseded in places
    │                                 #   (describes model/pipeline.py and model/citation_checker.py
    │                                 #   as the eventual home for logic that later moved to pipeline/
    │                                 #   and rules/ respectively) - kept for historical context only
    └── Phase Explanation.md           # earliest project plan - meaningfully stale (WordToken,
                                      #   Redis, scripts/-based corpus ingestion) - kept as a historical
                                      #   record of the original plan, not a current reference

---

## notes on this listing vs. the repo

- `test_deps.py` and `test_gpu.py` sitting at the repo root (not in
  `scripts/` or `tests/`) look like ad hoc personal debugging scripts —
  worth moving into `scripts/` or removing if they're no longer needed.
- `main.py` at the root is unmodified `uv init` boilerplate
  (`print("Hello from nyayai!")`) — not used by anything, safe to delete.
- DVC (`.dvc/`, `.dvcignore`, `data.dvc`) is present but not documented
  anywhere else in the repo (README, other docs) — worth either
  documenting what it tracks and how to use it, or removing it if it's
  not actually in active use.
- `docker-compose.yml` still defines a `redis` service. Nothing in the
  current codebase uses it — Celery uses the filesystem broker + SQLite
  result backend (see `workers/celery_app.py` and `config/settings.py`).
  This is a known, tracked inconsistency, not intentional.