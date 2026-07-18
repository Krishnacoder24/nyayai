"""
runs eval on data/training/test.jsonl using whatever's in model/checkpoint/,
prints a per-label precision/recall/F1 report via seqeval.

this is a SEPARATE inference loop from model/predict.py on purpose -
predict.py operates on Chunk objects from model/preprocess.py's
sliding-window document pipeline (built for a whole real document, with
token_to_span mapping back to LineSpans). test.jsonl's examples are
already short, pre-chunked training examples with no LineSpans involved -
reusing predict.py's machinery here would mean fighting its shape instead
of just running the model.
"""

import json
import logging

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from train.dataset import load_examples, encode_example
from train.metrics import align_predictions, full_report
from train.collator import build_collator

from config.settings import settings

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = settings.checkpoint_dir
TEST_JSONL = "data/training/test.jsonl"
BATCH_SIZE = 16


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model = AutoModelForTokenClassification.from_pretrained(CHECKPOINT_DIR)
    model.to(device)
    model.eval()

    examples = load_examples(TEST_JSONL)
    encoded = [encode_example(ex["words"], ex["labels"], tokenizer) for ex in examples]
    logger.info(f"evaluating on {len(encoded)} test examples")

    collator = build_collator(tokenizer)

    all_true, all_pred = [], []

    with torch.no_grad():
        for i in range(0, len(encoded), BATCH_SIZE):
            batch = collator(encoded[i:i + BATCH_SIZE])
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits.cpu().numpy()
            label_ids = batch["labels"].cpu().numpy()

            true_labels, pred_labels = align_predictions(logits, label_ids)
            all_true.extend(true_labels)
            all_pred.extend(pred_labels)

    report = full_report(all_true, all_pred)
    print(report)

    with open("data/training/eval_report.txt", "w") as f:
        f.write(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()