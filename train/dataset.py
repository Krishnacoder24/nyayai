"""
loads the JSONL files scripts/generate_data.py produces (word lists +
word-level BIO labels) and turns them into tokenized, subword-aligned
examples ready for the HuggingFace Trainer.

label alignment follows the same PRINCIPLE as model/preprocess.py's
inference-time alignment - only the FIRST subword of each word gets a real
label, every continuation subword is masked out - but the MECHANISM differs
because the purpose differs:
  - preprocess.py (inference) marks continuations with None, so
    postprocess.py knows to skip them when mapping predictions back to
    source LineSpans
  - this file (training) marks continuations with -100, which is
    CrossEntropyLoss's reserved "ignore this position" index - so the model
    is never penalized for whatever it predicts on a continuation subword

getting this backwards (e.g. repeating a word's label across every one of
its subwords) was a real bug caught during preprocess.py's development -
the same failure mode applies here if this isn't careful about it.
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

from model.schemas import LABEL2ID
from config.settings import settings

IGNORE_INDEX = -100
MAX_LENGTH = 512


def load_examples(jsonl_path: Path) -> list[dict]:
    examples = []
    with open(jsonl_path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


class TokenClassificationDataset:
    """
    wraps a list of {"words": [...], "labels": [...]} examples, tokenizing
    and aligning labels lazily via __getitem__ - this is enough surface
    (len + indexing) for a torch DataLoader without needing the separate
    `datasets` library.
    """

    def __init__(self, examples: list[dict], tokenizer=None):
        self.examples = examples
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(settings.bert_checkpoint)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return encode_example(example["words"], example["labels"], self.tokenizer)


def encode_example(words: list[str], word_labels: list[str], tokenizer) -> dict:
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    word_ids = encoding.word_ids()
    aligned_labels = []
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(IGNORE_INDEX)  # [CLS]/[SEP]/padding
        elif word_id != previous_word_id:
            aligned_labels.append(LABEL2ID[word_labels[word_id]])  # first subword of a word
        else:
            aligned_labels.append(IGNORE_INDEX)  # continuation subword
        previous_word_id = word_id

    encoding["labels"] = aligned_labels
    return encoding


def load_dataset(jsonl_path: Path, tokenizer=None) -> TokenClassificationDataset:
    return TokenClassificationDataset(load_examples(jsonl_path), tokenizer)