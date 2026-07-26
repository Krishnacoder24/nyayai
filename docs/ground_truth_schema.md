# Real-document ground truth format

For each real document you want to hand-label, create a `.json` file
with the **same filename stem** as the PDF, in the same directory:

```
data/real_eval/
├── sample_fir_01.pdf
├── sample_fir_01.json
├── sample_contract_03.pdf
├── sample_contract_03.json
```

Each `.json` file is a list of error objects:

```json
[
  {
    "text": "Section 302 of the Indian Penal Code",
    "type": "CITE",
    "note": "document dated 2025-03-12, should cite BNS 103"
  },
  {
    "text": "accussed",
    "type": "SPELL",
    "note": "misspelling of 'accused'"
  },
  {
    "text": "the accused was arrest on 5th June",
    "type": "GRAM",
    "note": "should be 'was arrested'"
  }
]
```

**Fields:**
- `text` (required) — the exact substring, copy-pasted straight out of
  the document, containing the error. Matching is case-insensitive and
  whitespace-normalized, but otherwise exact — don't paraphrase it.
- `type` (required) — one of `GRAM`, `CITE`, `SPELL` (matching the
  model's actual label set, not the `ENTITY` category — entity
  consistency is checked by `rules/entity_checker.py` separately, not
  by this model, so it has no place in this ground truth either).
- `note` (optional) — free text explaining the error, for your own
  reference when reviewing results. Not used in scoring.

**Why match on exact text instead of character/word offsets:** a human
labeler can just copy-paste the offending phrase directly, without
needing to agree with the model's internal tokenization on where an
error "starts" and "ends" - avoids an entire class of false mismatches
that have nothing to do with whether the model actually caught the
error.

**How many documents is "enough"?** Even 10-20 real documents, each
with a handful of real errors you've spotted by eye, is enough to tell
you whether the model is doing something real or just pattern-matching
the synthetic corruption functions it was trained on - it doesn't need
to be a large benchmark to be a genuinely useful sanity check.