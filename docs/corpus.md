# corpus

## overview

the corpus layer ingests raw Indian legal text (IPC, BNS, BNSS,
Constitution, CPC) into Qdrant so `rules/citation_checker.py` can verify
whether a cited section actually exists and is currently active.

this is a one-time setup step (or re-run when acts are amended). once
ingested, citation checking works in real time without any model
loading — pure exact-field lookup against Qdrant.

**status: infrastructure is fully built (schemas, chunker, embeddings,
uploader, search), but only IPC has an actual parser, and it's still the
original naive regex version — not the TOC-guided rewrite. BNS, BNSS,
CPC, and Constitution parsers are 0-byte placeholder files.** Everything
below describes the real, current schema and pipeline — treat the
"acts ingested" table as target state, not current state.

---

## acts (target coverage)

| act | full name | status | replaces | parser exists? |
|---|---|---|---|---|
| IPC | Indian Penal Code, 1860 | partially repealed | — | yes, but old/naive version |
| BNS | Bharatiya Nyaya Sanhita, 2023 | active | replaces IPC | no — 0-byte file |
| CrPC | Code of Criminal Procedure, 1973 | repealed | — | not tracked separately |
| BNSS | Bharatiya Nagarik Suraksha Sanhita, 2023 | active | replaces CrPC | no — 0-byte file |
| CPC | Code of Civil Procedure, 1908 | active | — | no — 0-byte file |
| Constitution | Constitution of India, 1950 | active | — | no — 0-byte file |

**critical distinction:** BNS ≠ BNSS.
- BNS (Bharatiya Nyaya Sanhita) replaces the IPC — it's the substantive
  criminal law (what acts are crimes, what punishments apply)
- BNSS (Bharatiya Nagarik Suraksha Sanhita) replaces CrPC — it's the
  procedural criminal law (how cases are investigated, tried, appealed)

a document citing "Section 302 IPC" after July 2024 (when BNS came into
force) is citing a repealed provision. the corpus is meant to flag this —
but this only works today for IPC citations, since IPC is the only act
currently ingestible, and even that parser hasn't had its TOC-guided
rewrite land yet.

---

## sources

raw legal text is sourced from:
- **IndiaCode** (indiacode.nic.in) — official government source for IPC,
  BNS, BNSS, CPC, Constitution PDFs. authoritative, frequently updated.
- **Legislative.gov.in** — gazette notifications for amendments

raw files live in `corpus/sources/`, which is gitignored (too large).

---

## ingestion pipeline (as actually implemented)

```
corpus/sources/{act}/raw PDF
          │
          ▼
corpus/parser.py  ->  corpus/parsers/{act}.py
          │  dispatches to the registered act-specific parser class
          │  (currently only "IPC" is registered in _PARSERS — see below)
          │  output: list[Section]
          │
          ▼
corpus/chunker.py
          │  chunk_section(section) -> list[Passage]
          │  splits by LEGAL STRUCTURE, not token count: the main
          │  operative text becomes one Passage, and each
          │  Explanation/Illustration/Exception becomes its own Passage
          │  output: list[Passage]
          │
          ▼
corpus/embeddings.py
          │  PassageEmbedder wraps InLegalBERT (hardcoded, not a choice
          │  between models — see "embedding model" below)
          │  uses the [CLS] token's final hidden state as the passage
          │  vector (768-dim)
          │  output: list[list[float]]
          │
          ▼
corpus/uploader.py
          │  upload_passages() pushes to Qdrant with a payload dict
          │  output: point count
          │
          ▼
Qdrant collection: "legal_corpus"
```

`corpus/parser.py` is a thin dispatch layer, not a parser itself:

```python
_PARSERS = {
    "IPC": IPCParser(),
    # "BNS": BNSParser(),
    # "BNSS": BNSSParser(),
    # "CPC": CPCParser(),
    # "CONSTITUTION": ConstitutionParser(),
}
```
The commented-out lines are exactly what's in the file today — adding a
new act means writing `corpus/parsers/{act}.py` and uncommenting its line
here, nothing else changes.

There is deliberately **no shared base class across parsers** — each
act's real-world PDF has different formatting quirks, so parsers are
independent (resolved in issue #26: `corpus/parsers/base.py`'s unused
`ChapterSectionParser` was deleted rather than adopted). Parsers do
still share plain PDF-reading helpers — page extraction and
running-header stripping — via `corpus/pdf_utils.py`, since that's
generic PDF plumbing, not act-specific grammar; sharing a *function* is
not the same thing as inheriting a *class*.

---

## Section dataclass (real fields — see `corpus/schemas.py`)

```python
@dataclass
class Section:
    act: str            # "IPC", "BNS", "BNSS", "CPC", "Constitution"
    unit_type: str       # "section" or "article"
    number: str          # "302", "304A", "21" - string to handle alphanumeric
    title: str
    body: str            # full section/article text
    status: str          # "active" or "repealed"
    metadata: dict = field(default_factory=dict)
    # act/parser-specific extras live here: chapter, part, effective_date,
    # replaced_by, etc. - a dict rather than named fields so each parser can
    # attach what's relevant to its act without changing this schema
```

Note the field names: `number` (not `section_no`), `body` (not `text`),
and no top-level `replaced_by`/`effective_date` fields — those live inside
the generic `metadata` dict instead, per-act, rather than being part of
the fixed schema. This was a deliberate design choice to let each parser
attach whatever extra fields its act needs (the Constitution has `part`
and `article`, IPC/BNS have `chapter`) without changing the shared
dataclass.

## Passage dataclass (real fields)

```python
@dataclass
class Passage:
    act: str
    unit_type: str
    number: str
    title: str
    status: str
    text: str            # this passage's chunk of text (subset of Section.body)
    metadata: dict = field(default_factory=dict)
    # copy of the parent Section's metadata, plus a "part" key describing
    # which structural piece this passage is: "body", "explanation_1",
    # "illustration_2", "exception", etc.
```

Every `Passage` fully repeats its parent `Section`'s identifying fields
(`act`, `number`, `title`, `status`) so a search hit is self-contained and
never needs a join back to the original `Section` — this matters because
Qdrant only stores the `Passage`, not the `Section`.

---

## how chunking actually works

`corpus/chunker.py`'s `chunk_section()` does **not** split by token count
or a fixed window. It splits on structural markers in the legal text —
`Explanation.—`, `Illustration (a).—`, `Exception.—`, matched via:

```python
STRUCTURAL_MARKER = re.compile(
    r'\n\s*(Explanation|Illustration|Exception)\s*(\d*)\s*\.\s*[-—–]\s*',
    re.IGNORECASE,
)
```

The text before the first marker becomes one `Passage` tagged `part:
"body"`. Everything between two markers (or from the last marker to the
end) becomes its own `Passage`, tagged with a `part` like `explanation_1`
or `illustration_a`. If a section has no structural markers at all, the
whole body becomes a single `Passage`.

This keeps every chunk semantically whole — an Explanation is a complete
legal thought — instead of an arbitrary word-count window that might cut
a sentence in half. This is a deliberate departure from the token-window
chunking approach used elsewhere in the project (`model/preprocess.py`
does use sliding-window chunking, but that's a different problem: fitting
long documents into BERT's 512-token limit for inference, not producing
retrieval-quality passages).

---

## Qdrant collection schema (as actually implemented)

**collection name:** `legal_corpus` (`corpus.uploader.COLLECTION_NAME`,
also mirrored in `settings.qdrant_collection`)

**vector size:** 768 dimensions (matches InLegalBERT's hidden size —
`corpus/uploader.py`'s `VECTOR_SIZE = 768`)

**payload fields per point** (from `upload_passages()` in `corpus/uploader.py`):

```json
{
  "act": "IPC",
  "unit_type": "section",
  "number": "302",
  "title": "Murder",
  "status": "repealed",
  "text": "Whoever commits murder shall be punished...",
  "metadata": {
    "chapter": "XVI",
    "part": "body"
  }
}
```

Note there is no top-level `replaced_by` or `effective_date` in the
payload — if a parser sets those, they'd live inside `metadata`, but as
of today no parser actually populates a `replaced_by` value anywhere; the
verified IPC→BNS conversion table this would come from doesn't exist yet
as real data (see below).

**indexes:** `number` and `act` are indexed as keyword fields
(`PayloadSchemaType.KEYWORD`) for exact-match filtering — this is what
makes citation lookup fast. Collection creation and index setup both
happen automatically inside `ensure_collection()`, called once on first
ingest; no manual Qdrant schema setup is needed.

---

## how `rules/citation_checker.py` queries Qdrant

Citation checking never touches the Qdrant client directly — it goes
through `corpus.search.lookup_section()`, which does an exact-field
filter, no vector similarity:

```python
def lookup_section(number: str, act: str, client=None) -> dict | None:
    result = client.query_points(
        collection_name="legal_corpus",
        query_filter=Filter(
            must=[
                FieldCondition(key="number", match=MatchValue(value=number)),
                FieldCondition(key="act", match=MatchValue(value=act)),
            ]
        ),
        limit=1,
        with_payload=True,
    )
    return result.points[0].payload if result.points else None
```

If `lookup_section()` returns `None` → the section doesn't exist in the
corpus → flag as a citation error. If found but `payload["status"] ==
"repealed"` → flag as a citation error, ideally with a suggestion pointing
at `payload["metadata"].get("replaced_by")` — though as noted above, no
parser currently populates that key with real data.

**Known inconsistency:** `corpus.uploader.get_client()` (and therefore
`corpus.search.lookup_section()`'s default) hardcodes
`url="http://localhost:6333"` rather than reading `settings.qdrant_url` —
so changing `QDRANT_URL` in `.env` currently has no effect on ingestion or
search unless a client is explicitly passed in. This is a tracked bug, not
intentional behavior.

---

## IPC → BNS conversion

Some well-known mappings, for reference only — **not yet stored as real
ingested data anywhere in the corpus**:

| IPC section | offence | BNS equivalent | notes |
|---|---|---|---|
| 302 | Murder | 103 BNS | |
| 304A | Causing death by negligence | 106 BNS | |
| 307 | Attempt to murder | 109 BNS | |
| 376 | Rape | 64 BNS | expanded scope |
| 420 | Cheating | 318 BNS | |
| 498A | Cruelty by husband | 85 BNS | |

**important:** do not fabricate or assume conversion mappings beyond the
handful above, which are well-documented public knowledge. A complete,
verified IPC→BNS (and CrPC→BNSS) mapping table needs to be sourced from
the official BNS text (indiacode.nic.in) or the MHA comparative statement,
stored as a real versioned data file, and wired into whichever parser
attaches `replaced_by` metadata to repealed sections. Wrong mappings in
the corpus would cause the citation checker to suggest incorrect
corrections — worse than not suggesting anything at all. This table does
not exist as data in the repo today; the table above is illustrative
documentation only.

---

## embedding model

`corpus/embeddings.py`'s `PassageEmbedder` is **hardcoded to InLegalBERT**
(`law-ai/InLegalBERT`, via `config.constants.MODEL_NAME`) — this is not a
runtime choice between two options. It uses the `[CLS]` token's final
hidden state as the passage embedding, the standard trick for pulling a
sentence-level embedding out of a BERT-family model that wasn't
specifically trained with a pooling objective:

```python
outputs = self.model(**inputs)
cls_vectors = outputs.last_hidden_state[:, 0, :]  # [CLS] token per sequence
```

For citation checking specifically (exact field lookup by `number` +
`act`), the embedding model doesn't actually matter — vectors are stored
but never queried by similarity for this feature. The 768-dim vector
exists to support a future semantic-search feature ("find the BNS
equivalent of this IPC section" via similarity, not exact lookup), which
is not yet built.

**Note:** the file's own top-of-file comment currently says
`MODEL_NAME = constants.MODEL_NAME  # "nlpaueb/legal-bert-base-uncased"` —
that comment is stale; the real value resolved from
`config/constants.py` is `"law-ai/InLegalBERT"`, consistent with the rest
of the project. There's also an unused `from asyncio import constants`
import left in the file above the real `from config import constants`
import — harmless since the second import shadows it, but worth cleaning
up.

---

## running ingestion

```bash
uv run python scripts/ingest_corpus.py --act ipc
uv run python scripts/ingest_corpus.py --all
```

(`--act bns`, `--act bnss`, etc. will work once those parsers exist and
are registered in `corpus/parser.py`'s `_PARSERS` dict — today only
`--act ipc` will actually succeed.)

**Qdrant must be running first:**
```bash
docker-compose up -d qdrant
```
default ports: REST API `http://localhost:6333`, gRPC `localhost:6334`.
the collection is created automatically on first ingest if it doesn't
exist — no manual schema setup needed.

---

## verifying the corpus

after ingestion, spot-check a few known sections:

```bash
uv run python -c "
from corpus.search import lookup_section
print(lookup_section('302', 'IPC'))   # should show status=repealed (once IPC is fully re-parsed)
print(lookup_section('999', 'IPC'))   # should return None (doesn't exist)
"
```

---

## re-ingestion

when an act gets amended (sections added, renumbered, repealed):

1. download the updated act PDF from IndiaCode
2. drop the existing points for that act via `corpus.uploader.drop_act(client, act)`
   (removes only that act's points; the rest of the collection is untouched)
3. re-run `scripts/ingest_corpus.py --act {act}`
4. spot-check with the verification snippet above
