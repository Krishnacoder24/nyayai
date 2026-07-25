# NyayAI Architecture Improvements

This document lists the remaining architectural improvements before the project reaches a stable, production-ready backend. These are **refinements**, not redesigns. The overall architecture is already strong, and after these changes the focus should shift almost entirely to implementation and testing.

---

# Priority 1 (Before Building the API)

These are the only architectural changes I consider important before exposing the project through FastAPI.

---

## 1. Extract ML Inference into `model/inference.py`

### Current

```
engine.py
    └── _run_ml()
            ├── build_chunks()
            ├── predict()
            └── build_error_spans()
```

### Problem

`engine.py` currently knows how the ML pipeline works internally.

Today it is only three function calls.

Tomorrow it may include:

- model selection
- confidence thresholds
- ONNX inference
- batching
- GPU warm-up
- multiple checkpoints

The orchestrator should never contain ML implementation details.

### Recommendation

Create:

```
model_runnere inside pipeline
```

Expose only:

```python
analyze(spans: list[LineSpan]) -> list[ErrorSpan]
```

Internally:

```
LineSpans
    ↓
build_chunks()
    ↓
predict()
    ↓
build_error_spans()
    ↓
ErrorSpans
```

Then `engine.py` simply becomes:

```python
ml_errors = model.analyze(spans)
```

The pipeline knows **what** happens, not **how** it happens.

---

## 2. Make Rule Engine Pluggable

### Current

```python
citation_errors = check_citations(spans)
entity_errors = check_entities(spans)
```

Every new checker requires editing `engine.py`.

Eventually this becomes:

```python
check_dates(...)
check_formatting(...)
check_abbreviations(...)
check_cross_references(...)
...
```

### Recommendation

Instead expose a registry.

```python
RULES = [
    CitationChecker(),
    EntityChecker(),
]
```

or

```python
RULES = [
    check_citations,
    check_entities,
]
```

Execution becomes:

```python
errors = []

for rule in RULES:
    errors.extend(rule.check(spans))
```

Adding a new rule only requires registering it.

The pipeline remains unchanged.

---

## 3. Introduce `AnalysisService`

Currently multiple entry points will eventually duplicate the same workflow.

```
FastAPI
    ↓
OCR
    ↓
Pipeline
```

```
CLI
    ↓
OCR
    ↓
Pipeline
```

```
Celery
    ↓
OCR
    ↓
Pipeline
```

### Recommendation

Introduce

```
services/
    analysis.py
```

```python
class AnalysisService:

    def analyze(pdf_path):
        ...
```

Internally:

```
PDF
    ↓
OCR
    ↓
Pipeline
    ↓
Renderer
    ↓
Report
```

Everything else calls this service.

- FastAPI
- CLI
- Celery
- Tests

Only one implementation exists.

---

## 4. Move Remaining Thresholds into Configuration

Current:

```python
OVERLAP_THRESHOLD = 0.5
```

This should become

```
config/constants.py
```

```python
DEDUPLICATION_IOU_THRESHOLD
```

The project already centralizes almost every constant.

This should follow the same convention.

---

## 5. Add Error Provenance

`ErrorSpan` currently contains no information about where the error originated.

Example:

```python
source: str
```

Possible values:

```
ml
citation_rule
entity_rule
date_rule
formatting_rule
```

Benefits:

- easier debugging
- frontend badges
- evaluation metrics
- future ensemble weighting
- explainability

---

# Priority 2 (After MVP)

These changes improve extensibility but are not blockers.

---

## 6. Abstract the Embedding Layer

Currently the corpus is tied to InLegalBERT.

Instead expose a generic interface.

```python
class PassageEmbedder:
    embed(passages)
```

Possible implementations:

```
InLegalBERTEmbedder
BGEEmbedder
E5Embedder
```

The ingestion pipeline should never depend on a specific embedding model.

---

## 7. Parser Registry

Instead of maintaining a static parser dictionary forever,

consider exposing:

```python
register_parser(...)
```

This makes supporting future Acts easier.

Current implementation is acceptable.

---

## 8. Configurable Chunkers

Structural markers are currently hardcoded:

```
Explanation
Illustration
Exception
```

Eventually each parser should expose its own structural markers.

Example:

```python
STRUCTURAL_MARKERS = [
    ...
]
```

This allows Acts with different legal structures to define their own chunk boundaries.

---

## 9. Typed Metadata (If Needed)

Current:

```python
metadata: dict
```

This is flexible and should remain.

However, if certain fields become universally used, such as:

- chapter
- part
- schedule

they should eventually become first-class schema fields.

Do **not** change this prematurely.

Monitor how the metadata evolves first.

---

# Priority 3 (Scaling)

These improvements become useful only after the system is stable.

---

## Parallel Rule Execution

Current execution:

```
ML
    ↓
Citation
    ↓
Entity
```

Future:

```
ML ─────────┐
            │
Citation ───┼── Merge
            │
Entity ─────┘
```

Rule engines are independent and can eventually execute concurrently.

---

## Sorting Layer

Reading-order sorting is currently inside `pipeline/engine.py`.

Since sorting is presentation-related rather than analysis-related, it could eventually move into the renderer layer.

This is a very low-priority refinement.

---

## Interface-Based Architecture

Long-term every subsystem should expose a stable interface.

Examples:

```
Parser

Chunker

Embedder

Uploader

Searcher

RuleChecker
```

rather than free functions.

This improves testing and future extensibility but is unnecessary for the MVP.

---

# Architecture Decisions That Should NOT Change

These decisions are already strong and should remain.

- Single Qdrant collection (`legal_corpus`)
- Parser hierarchy
- `Section → Passage` domain model
- Search abstraction
- Pipeline orchestration
- Rule-based + ML hybrid design
- OCR output (`LineSpan`)
- Merge layer
- Deduplication strategy

No redesign is recommended in these areas.

---

# Development Roadmap

## Before API

- [ ] Move ML inference into `model/inference.py`
- [ ] Introduce pluggable rule registry
- [ ] Implement `AnalysisService`
- [ ] Move remaining thresholds into configuration
- [ ] Add provenance (`source`) to `ErrorSpan`

---

## After MVP

- [ ] Abstract embedding models
- [ ] Parser registry
- [ ] Configurable chunking
- [ ] Promote commonly used metadata fields (only if needed)

---

## Future Scaling

- [ ] Parallel rule execution
- [ ] Better embedding models
- [ ] Hybrid retrieval
- [ ] Distributed workers
- [ ] ONNX inference

---

# Final Assessment

The architecture has reached the point where further redesign will provide very little value.

The remaining work is primarily implementation.

The next major insights will come from:

- parsing real legal documents
- ingesting the complete corpus
- running retrieval experiments
- evaluating model predictions
- testing end-to-end workflows

From this point onward, practical usage is likely to reveal more meaningful improvements than additional architectural discussions.