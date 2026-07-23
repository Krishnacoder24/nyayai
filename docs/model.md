# model

**status:** the inference scaffold described below (preprocess → predict →
postprocess) is fully built and matches the real code in `model/`. The
fine-tuning section further down describes the *planned* training
configuration — `train/` is scaffolded but has not actually been run yet,
so there is no fine-tuned checkpoint, no real F1 numbers, and
`model/predict.py` currently returns all-`O` labels (no detected errors)
for every document. Citation and entity checking are unaffected by this —
they're pure rule-based checkers in `rules/`, not part of this ML path.

## overview

the ML component of nyayai uses `law-ai/InLegalBERT` — a BERT model
pretrained on Indian legal text — fine-tuned as a token classifier to
detect spelling errors, grammar errors, and wrong citations at the token
level across Indian legal documents.

the model is one part of the error detection system. rule-based checkers
in `rules/` handle citation lookup and entity consistency independently.
the model's job is specifically: given a sequence of legal text tokens,
classify each one into a BIO error label.

---

## InLegalBERT

**model:** `law-ai/InLegalBERT`
**architecture:** BERT-base (12 layers, 768 hidden, 12 attention heads, 110M params)
**pretrained on:** Indian Supreme Court and High Court judgments, IPC, BNS,
Constitution, legal articles — domain adapted from bert-base-uncased

**why InLegalBERT over generic BERT:**
generic BERT doesn't know "IPC", "FIR", "cognizable", "magistrate",
"petitioner", "respondent", "Bharatiya Nyaya Sanhita" as coherent legal
concepts. InLegalBERT's pretraining on Indian legal corpora means its
embeddings already encode legal vocabulary and citation patterns. this
gives better token representations as a starting point for fine-tuning
compared to starting from generic English BERT.

**what it does NOT come with:**
InLegalBERT's published fine-tuned heads are for:
- Legal Statute Identification (multi-label classification)
- Rhetorical Role Segmentation (7 functional document parts)
- Court Judgment Prediction (outcome classification)

none of these are error detection. the token classification head for
spelling/grammar/citation errors is trained by us on synthetic data.

---

## label scheme

BIO (Beginning-Inside-Outside) encoding for token classification.

```
O         no error — token is correct
B-SPELL   first token of a spelling error span
I-SPELL   continuation token of a spelling error span
B-GRAM    first token of a grammar error span
I-GRAM    continuation token of a grammar error span
B-CITE    first token of a wrong citation span
I-CITE    continuation token of a wrong citation span
B-ENT     first token of an entity inconsistency span
I-ENT     continuation token of an entity inconsistency span
```

**total labels: 9**

```python
LABELS = ["O", "B-SPELL", "I-SPELL", "B-GRAM", "I-GRAM",
          "B-CITE", "I-CITE", "B-ENT", "I-ENT"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL  = {i: l for i, l in enumerate(LABELS)}
```

**why BIO over flat labels:**
flat labels ("SPELL", "GRAM", "CITE") can't represent multi-token spans.
"Section 302 IPC" is 3 tokens — with flat labels you'd get three separate
single-token errors. BIO encoding lets postprocess.py reconstruct
"Section 302 IPC" as one span with one merged bbox. B- marks the start,
I- continues it, O or a different B- ends it.

**why ENT is in the model labels:**
entity checker in rules/ handles most entity inconsistencies via NER +
fuzzy matching. but subtle entity errors embedded in running text (not
extracted as NER entities) may be caught by the model if trained on
synthetic ENT examples. both can fire independently — deduplication in
pipeline/ handles overlap.

---

## inference pipeline

```
list[LineSpan]
      │
      ▼
model/preprocess.py
      │  groups lines into 512-token chunks
      │  keeps token -> LineSpan bbox mapping
      │
list[Chunk]
      │
      ▼
model/predict.py
      │  loads InLegalBERT from model/checkpoint/
      │  pads chunks to longest in batch
      │  runs forward pass (batch_size=8)
      │  softmax -> argmax -> label IDs
      │
list[list[int]]  (label ID per token, per chunk)
      │
      ▼
model/postprocess.py
      │  reads BIO sequences
      │  looks up source LineSpan via token_to_span
      │  merges bboxes across span tokens
      │  confidence = mean softmax prob across span tokens
      │
list[ErrorSpan]
```

---

## preprocess.py

**problem:** InLegalBERT (BERT-base) has a hard 512 subword token limit.
a legal document has hundreds of lines — far more than 512 tokens.

**solution:** sliding window chunking with overlap.

```
window = 510 tokens  (512 - 2 for [CLS] and [SEP])
stride = 128 tokens  (overlap between consecutive chunks)
```

a 600-token document becomes:
```
chunk 0: tokens 0-509    (+ [CLS], [SEP])
chunk 1: tokens 382-591  (128 token overlap with chunk 0)
```

the overlap ensures that spans near chunk boundaries appear fully in
context in at least one chunk. without overlap, a citation like
"Section 302 IPC" split across a boundary would be seen as fragments.

**subword alignment:**
BERT tokenizes into subwords. "IPC" may become ["IP", "##C"].
only the first subword of each word gets a real `token_to_span` index.
continuation subwords get `None` and are ignored by postprocess.py.
this is the standard HuggingFace convention for token classification —
`word_ids()` from the fast tokenizer handles this.

```python
# word_ids() output for ["Section", "302", "IPC"]
# where "IPC" -> ["IP", "##C"]
[0, 1, 2, 2]
#             ^ same word_id = subword continuation
```

only positions where `word_id != prev_word_id` get a span index.
continuations get `None`.

**Chunk dataclass:**
```python
@dataclass
class Chunk:
    input_ids: list[int]        # token IDs including [CLS] and [SEP]
    attention_mask: list[int]   # 1 for real tokens, 0 for padding
    token_to_span: list[int | None]  # span index or None for ignored positions
    span_indices: list[int]     # which LineSpan indices this chunk covers
```

---

## predict.py

**checkpoint detection:**
looks for `model/checkpoint/config.json` on startup.
- found → loads fine-tuned weights, real predictions
- not found → logs warning, returns all `O` labels (no errors)

this means the full pipeline runs without crashing before fine-tuning.
citation and entity errors still work (they don't use the model).
ML errors become real the moment weights are dropped into checkpoint/.

**batching:**
chunks from one document are batched together for inference.
batch_size=8 is the default for 6GB VRAM (inference only, no gradients).
if VRAM is tight, drop to 4. if you have headroom, try 16.

**dynamic padding:**
chunks in a batch have different lengths (last chunk is almost always
shorter). we pad to the longest in each batch, not always to 512.
a batch of 8 short chunks doesn't get padded to 512 unnecessarily.
attention_mask=0 on pad positions tells the model to ignore them.

**why `model.eval()` and `torch.no_grad()` both matter:**
- `model.eval()` — disables dropout layers (stochastic during training,
  deterministic at inference). forgetting this gives different predictions
  on every run for the same input.
- `torch.no_grad()` — stops pytorch building the computation graph for
  backprop. without it, pytorch tracks every operation for gradient
  computation even though we never call .backward(). wastes ~30-40% of
  VRAM at inference for nothing.
both are required. neither is optional.

---

## postprocess.py

**BIO decoding:**
reads label IDs token by token. when it sees `B-X`, starts a new span.
`I-X` of the same type extends it. `O` or a different `B-` flushes the
current span and starts fresh (or nothing, for `O`).

**orphan I- recovery:**
a well-trained model shouldn't emit `I-CITE` without a preceding `B-CITE`.
base model with random head absolutely will. we treat orphan `I-` as `B-`
and start a new span rather than crashing or silently dropping it.

**bbox merging:**
a span may cover multiple LineSpans (e.g. a citation that wraps a line
boundary). we collect all contributing LineSpan bboxes and merge:
```
x0 = min(all x0s)
y0 = min(all y0s)
x1 = max(all x1s)
y1 = max(all y1s)
```
gives a tight box enclosing the full span across all contributing lines.

**confidence:**
mean softmax probability across all tokens in the span.
```
confidence = mean(softmax(logits)[token_pos][predicted_label_id])
             for all token positions in the span
```
stored on ErrorSpan. frontend can threshold on this — e.g. only show
errors with confidence > 0.7.

---

## fine-tuning (train/) — planned configuration, not yet executed

Everything in this section describes the scaffolded design in `train/` —
`dataset.py`, `collator.py`, `train.py`, `metrics.py`, `evaluate.py` all
exist and are wired together correctly, but no training run has happened
yet. Treat the hyperparameters below as the intended starting point, not
as numbers validated by an actual run. Once training runs, this section
should be updated with the real F1 achieved and any hyperparameters that
changed during tuning.

### training data
no existing public dataset for Indian legal error detection exists.
we generate synthetic data:
1. take real correct Indian legal text (IPC/BNS/Constitution + judgments)
2. programmatically corrupt it:
   - SPELL: swap characters, delete characters, phonetic substitutions
   - GRAM: wrong verb form, agreement errors, missing articles
   - CITE: replace valid section numbers with invalid/repealed ones
   - ENT: introduce name variants (Ramesh → Rakesh, Sheikh → Shaikh)
3. generate (corrupted_text, BIO_labels) pairs
4. store as `data/training/train.jsonl`, `val.jsonl`, `test.jsonl`

**important:** GRAM corruption must run before SPELL/CITE in the
corruption pipeline. grammar changes alter token count and would
invalidate index-based BIO tags applied afterward.

### training setup
```
base model:   law-ai/InLegalBERT
head:         AutoModelForTokenClassification(num_labels=9)
optimizer:    AdamW, lr=2e-5
scheduler:    linear warmup (10% steps), linear decay
batch size:   16 (gradient accumulation if VRAM limited)
epochs:       5 (early stopping on val F1)
fp16:         True (halves VRAM, minimal accuracy impact)
metric:       seqeval span-level F1 (not token accuracy)
checkpoint:   saved on best val F1
```

### why span-level F1 not token accuracy
token accuracy is misleading for BIO tasks. if 95% of tokens are `O`
(correct), a model that predicts `O` for everything gets 95% accuracy
but detects zero errors. seqeval F1 measures whether complete spans
(B- through end of I- sequence) were correctly identified — this is
what actually matters for the proofreading use case.

### collator
`DataCollatorForTokenClassification` from HuggingFace. pads sequences
to the longest in each batch, sets label=-100 for padding positions
so they don't contribute to loss.

---

## checkpoint structure

```
model/checkpoint/
├── config.json               model architecture config
├── tokenizer.json            fast tokenizer vocab + rules
├── tokenizer_config.json     tokenizer settings
├── special_tokens_map.json   [CLS], [SEP], [PAD], [UNK] mappings
├── vocab.txt                 full vocabulary (30k tokens)
└── model.safetensors         trained weights (~440MB)
```

the entire `model/checkpoint/` directory is gitignored. weights are
not committed to the repo — too large, and they shouldn't be versioned
with source code. store them in a separate model registry or share via
direct download.

---

## adding a new label type

if you want to add a new error type (e.g. `B-DATE / I-DATE` for date
format errors):

1. add `"B-DATE"` and `"I-DATE"` to `LABELS` in `model/schemas.py`
2. add `"DATE": "date"` to `ERROR_TYPES` in `model/schemas.py`
3. add a highlight color in `ErrorSpan.highlight_color`
4. regenerate synthetic training data with DATE corruption examples
5. retrain from scratch (or fine-tune the existing checkpoint)
6. update `num_labels` in `predict.py` (auto-derived from `len(LABELS)`)

steps 1-3 take 5 minutes. steps 4-5 take hours. plan label additions
before training, not after.