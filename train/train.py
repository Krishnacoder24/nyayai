"""
fine-tunes InLegalBERT for token classification using the HuggingFace
Trainer API.

checkpoint loading/saving conventions here match model/predict.py exactly
on purpose: predict.py looks for a config.json in model/checkpoint/ and,
if found, loads BOTH the model and the tokenizer from that same directory.
so this file has to save the tokenizer into checkpoint_dir too, not just
the model weights - if it only saved model weights, predict.py would load
a tokenizer that doesn't necessarily match (e.g. after a future retrain
with a different base checkpoint).

hyperparameters below are reasonable BERT-fine-tuning defaults, not
empirically tuned for this specific task - this hasn't been run yet.
adjust batch size / gradient accumulation first if it doesn't fit in 6GB.
"""

import logging

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)

from model.schemas import LABELS, LABEL2ID, ID2LABEL
from train.dataset import load_dataset
from train.collator import build_collator
from train.metrics import compute_metrics

from config.settings import settings

logger = logging.getLogger(__name__)

BASE_CHECKPOINT = settings.bert_checkpoint
OUTPUT_DIR = settings.checkpoint_dir

TRAIN_JSONL = "data/training/train.jsonl"
VAL_JSONL = "data/training/val.jsonl"

# starting point, not yet empirically tuned - this project's known 6GB VRAM
# constraint (RTX 4050) is why batch size is small with gradient
# accumulation making up the effective batch size, same discipline as
# model/predict.py's INFERENCE_BATCH_SIZE and surya's chunked page batching
PER_DEVICE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4  # effective batch size = 8 * 4 = 32
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_CHECKPOINT)

    train_dataset = load_dataset(TRAIN_JSONL, tokenizer)
    val_dataset = load_dataset(VAL_JSONL, tokenizer)
    logger.info(f"train examples: {len(train_dataset)}, val examples: {len(val_dataset)}")

    model = AutoModelForTokenClassification.from_pretrained(
        BASE_CHECKPOINT,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # base model has no classification head yet - expected
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=True,  # halves memory vs full precision - matters at 6GB VRAM
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=build_collator(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # trainer.train() with load_best_model_at_end=True leaves the BEST
    # checkpoint (by eval f1) loaded in trainer.model, not just the last
    # epoch's - save_model() persists that best version
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)  # predict.py loads the tokenizer from here too
    logger.info(f"saved best checkpoint (by eval f1) to {OUTPUT_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()