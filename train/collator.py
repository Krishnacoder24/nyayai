"""
pads a batch of variable-length encode_example() outputs to the same
length - both input_ids/attention_mask (padded with the tokenizer's own
pad logic) and labels (padded with -100, so padding positions never
contribute to the loss, same reasoning as the continuation subwords in
dataset.py).

this is a thin wrapper, not custom logic - transformers' own
DataCollatorForTokenClassification already does exactly this correctly.
"""

from transformers import AutoTokenizer, DataCollatorForTokenClassification

from config.settings import settings


def build_collator(tokenizer=None) -> DataCollatorForTokenClassification:
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(settings.bert_checkpoint)
    return DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=-100)