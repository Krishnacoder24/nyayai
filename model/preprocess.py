"""
groups LineSpans into chunks that InLegalBERT can consume.

InLegalBERT (like all BERT models) has a hard 512 subword token limit.
a legal document can have hundreds of lines so we need to:
  1. pack lines into chunks that stay under 512 tokens
  2. keep a mapping from each token position back to its source LineSpan
     so postprocess.py can recover the bbox when a label is assigned

the token->span mapping is the critical piece. without it we'd know
"token 47 is a CITE error" but have no idea where on the page it is.
"""

from dataclasses import dataclass, field
from transformers import AutoTokenizer

from ocr.tokens import LineSpan


CHECKPOINT = "law-ai/InLegalBERT"
MAX_TOKENS = 512
STRIDE = 128  # overlap between chunks so spans at boundaries aren't missed


@dataclass
class Chunk:
    input_ids: list[int]
    attention_mask: list[int]
    token_to_span: list[int | None]  # index into original spans list, None for special tokens and subword continuations
    span_indices: list[int]          # which span indices are covered by this chunk


def build_chunks(spans: list[LineSpan]) -> list[Chunk]:
    """
    takes the full list of LineSpans from the OCR pipeline and returns
    a list of Chunk objects ready to be passed to predict.py.

    each chunk fits within MAX_TOKENS including [CLS] and [SEP].
    chunks overlap by STRIDE tokens so nothing at a boundary is missed.
    """
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    # step 1: tokenize each span individually and record which span
    # each token came from. we don't chunk yet, just flatten everything.
    all_token_ids = []
    all_token_to_span = []

    for span_idx, span in enumerate(spans):
        words = span.text.split()
        if not words:
            continue

        # is_split_into_words=True tells the tokenizer we're giving it
        # pre-split words, so it handles subword splitting per word and
        # word_ids() gives us back the word index for each token
        enc = tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=False,  # we add [CLS] and [SEP] per chunk below , cls  and sep stands for classification and separation tokens.
        )

        token_ids = enc.input_ids
        word_ids = enc.word_ids()  # list of ints, one per token, mapping back to word index

        prev_word_id = None
        for token_id, word_id in zip(token_ids, word_ids):
            all_token_ids.append(token_id)
            # word_id is None for special tokens (none here since add_special_tokens=False)
            # for subword continuations, word_ids repeats the same word index
            # we mark only the FIRST subword of each word with the span index,
            # continuations get None — postprocess ignores None positions
            if word_id is not None and word_id != prev_word_id:
                all_token_to_span.append(span_idx)
            else:
                all_token_to_span.append(None)
            prev_word_id = word_id


    # step 2: slide a window of (MAX_TOKENS - 2) over the flat token list-2 reserves space for [CLS] at start and [SEP] at end
    window = MAX_TOKENS - 2
    chunks = []
    i = 0

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    while i < len(all_token_ids):
        chunk_token_ids = all_token_ids[i:i + window]
        chunk_token_to_span = all_token_to_span[i:i + window]

        # wrap with [CLS] and [SEP], their token_to_span entries are None
        input_ids = [cls_id] + chunk_token_ids + [sep_id]
        token_to_span = [None] + chunk_token_to_span + [None]
        attention_mask = [1] * len(input_ids) #no padding 

        # track which span indices appear in this chunk so pipeline.py
        # can associate results back to the right spans
        span_indices = list({s for s in chunk_token_to_span if s is not None})

        chunks.append(Chunk(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_to_span=token_to_span, #type: ignore
            span_indices=span_indices,
        ))

        # advance by window - STRIDE so chunks overlap
        # if we're near the end and the remaining tokens fit in one chunk, stop
        next_i = i + window - STRIDE
        if next_i >= len(all_token_ids):
            break
        i = next_i

    return chunks