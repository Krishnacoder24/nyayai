# roadmap

## how this doc works now

This file used to be the single source of truth for what's next. As of
the repo audit that produced the 29-issue backlog, **the GitHub issue
tracker is now the living, authoritative roadmap** — 29 issues across 6
milestones (`M1 — Corpus` through `M6 — Ship It`), viewable at
`github.com/sahil3105kumar/nyayai/milestones`. This file stays useful as
a narrative summary of where the project has been and where it's headed,
but if the two ever disagree, the tracker wins — this file will drift the
way `docs/api.md` did before its rewrite, unless it's kept in sync
deliberately.

---

## current status (what's actually done)

**done, matching the frozen folder structure:**
- OCR pipeline (`ocr/`) — native + scanned PDF extraction, tested end to end
- model scaffold (`model/`) — preprocess/predict/postprocess wired
  correctly; no fine-tuned checkpoint yet, so ML detection currently
  returns nothing (this is intentional graceful degradation, not a bug)
- rule checkers (`rules/citation_checker.py`, `rules/entity_checker.py`)
- pipeline orchestration (`pipeline/engine.py`, `merger.py`, `deduplicate.py`)
- renderer (`renderer/annotate_pdf.py`, `report.py`, `html_report.py` —
  though `html_report.py` has a live crashing bug, tracked as a P0 issue)
- API + Celery workers (`api/`, `workers/`) — filesystem broker + SQLite,
  no Redis, no auth yet
- frontend (`frontend/`) — React + PDF.js, margin annotation rail, fully
  wired to the real API (not mock data)

**not done yet:**
- corpus act parsers — only IPC has a parser, and it's still the old
  naive version, not the TOC-guided rewrite; BNS/BNSS/CPC/Constitution are
  0-byte placeholders
- fine-tuning execution — `train/` is scaffolded but has never actually
  been run
- several confirmed bugs and cleanup items found during the full-repo
  audit (see the tracker's M2 and M4 milestones)
- a real automated test suite (most test files are currently `pass`
  stubs)
- deployment

---

## milestones (see the tracker for full issue text)

| Milestone | Focus |
|---|---|
| **M1 — Corpus** | All five act parsers (IPC rewrite + BNS/BNSS/CPC/Constitution), Qdrant URL default fix, verified IPC→BNS mapping table |
| **M2 — Model and Pipeline** | Fix the crashing HTML report bug, remove dead code (`model/pipeline.py`), model/tokenizer caching, run fine-tuning, pluggable rule registry, error provenance field |
| **M3 — Features** | Rich error-explanation tooltips, rule-based spelling/grammar complement |
| **M4 — Config and Housekeeping** | Settings cleanup, Redis removal (for real, everywhere), rewritten `.env.example`, full docs sync, dead stub cleanup |
| **M5 — Testing and Frontend Polish** | Real automated test suite, pagination bounds fix |
| **M6 — Ship It** | Final QA gate, then Vercel deployment (deployment is intentionally last) |

**Most parallelizable work:** the BNS/BNSS/CPC/Constitution parsers in M1
are independent of each other, and the `parsers/base.py` inheritance
question is resolved (issue #26 - no shared base class; deleted) —
multiple people can take these simultaneously.

**Pull forward regardless of milestone order:** the crashing HTML report
bug (M2) — it's a live P0 affecting every document with at least one
detected error, worth fixing immediately rather than waiting for M2 to
start in sequence.

---

## known limitations (snapshot — tracker has the current, itemized version)

**OCR:**
- surya OCR is slow (~10s per scanned page on RTX 4050). no fix planned —
  async processing via Celery hides the latency from the user.
- surya 0.9.3 is frozen — significantly behind current surya. upgrading
  requires vllm, which brings Docker + NVIDIA Container Toolkit
  dependencies. not worth it until 0.9.3 causes a real correctness
  problem.
- `transformers` pinned at `4.48.0` because newer versions break surya
  0.9.3's `SuryaOCRConfig` with `KeyError: 'encoder'`.

**entity checker:**
- `en_core_web_sm` handles Indian names poorly. "Ramesh Kumar" is not
  always detected as a PERSON entity. would benefit from a fine-tuned
  Indian legal NER model.
- greedy clustering misses some same-entity pairs where both forms are
  equally frequent.

**model:**
- no fine-tuned weights yet. ML error detection returns nothing until
  fine-tuning actually runs (tracked in M2).
- no correction suggestions. `ErrorSpan.suggestion` is empty for
  ML-detected errors; citations have suggestions where the corpus's
  `replaced_by` metadata exists. adding real corrections would need a
  seq2seq model.
- spelling checker via ML is overkill for simple typos — a rule-based
  spell checker with a custom legal dictionary is planned as a complement
  (tracked in M3).

**corpus:**
- IPC → BNS conversion mappings must be sourced from verified official
  documents only — no fabricated mappings. this slows down corpus
  building but wrong mappings would be worse than no suggestions.
- corpus is static — amendments after ingestion aren't reflected until
  re-ingestion.

**general:**
- no authentication on the API — fine for local use, must be added before
  any deployment (tracked in M6's deployment issue).
- no output cleanup — `data/uploads/` and `data/outputs/` accumulate
  indefinitely (tracked in M4).
- tested only on English-language legal documents — Hindi/regional
  language support is not planned in current scope.

---

## future ideas (not planned, just worth noting)

**correction suggestions:**
fine-tune a seq2seq model (mT5 or IndicBART) on (erroneous text, corrected
text) pairs to populate `ErrorSpan.suggestion` for spelling/grammar
errors. a significant separate project.

**semantic citation search:**
use vector similarity in Qdrant (the embedding infrastructure already
exists — see `docs/corpus.md`) to answer "what BNS section covers the
same offence as IPC 302?" — useful for documents that predate BNS and
need updating.

**multi-language support:**
IndicBERT or MuRIL for Hindi/regional language legal documents. surya OCR
already handles Hindi script.

**more rule checkers:**
```
rules/date_checker.py          date formats, logical consistency (date of
                               incident before date of FIR, etc.)
rules/cross_reference_checker.py   "as mentioned in paragraph 3" where
                               paragraph 3 doesn't exist (0-byte
                               placeholder exists already, see M2)
rules/formatting_checker.py    section numbering, header hierarchy
rules/signature_checker.py     signature/stamp fields missing
```

**document type detection:**
auto-detect document type (FIR, bail application, charge sheet, contract)
and apply type-specific rules.

**batch processing:**
process multiple documents in one API call — useful for law firms
processing a docket of related documents.

**comparison mode:**
compare two versions of the same document (draft vs. amended) and
highlight what changed, not just what's wrong.

---

## dependency decisions (frozen — verified against `pyproject.toml`)

| package | version | reason |
|---|---|---|
| surya-ocr | 0.9.3 | 0.20+ needs vllm, too heavy for dev setup |
| transformers | 4.48.0 | newer versions break surya's `SuryaOCRConfig` |
| torch | 2.4.1+cu124 | stable on RTX 4050, cu124 wheel confirmed working |
| qdrant-client | 1.17.1 | uses the `query_points` API, not the older `search` method |
| rapidfuzz | latest | no breaking changes in this library |
| spacy | latest | `en_core_web_sm` model version tied to spacy version |

---

## git branch history

```
main
 └─ feature/ocr        merged — OCR pipeline complete
 └─ feature/model       merged — model scaffold complete
 └─ feature/corpus      merged — infra complete, IPC parser only (naive version)
 └─ feature/renderer    merged
 └─ feature/api         merged
 └─ feature/frontend    merged — wired to the real API
```

All the phases from the original plan have merged. Remaining work is now
tracked as GitHub issues rather than future feature branches — see the
milestone table above.