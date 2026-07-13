"""
takes raw per-token label IDs from predict.py and reconstructs
ErrorSpans with real bboxes from the source LineSpans.

the core job:
  - read BIO labels token by token
  - when we see B-X, start a new span of type X
  - when we see I-X continuing the same type, extend it
  - when the span ends, look up the source LineSpans via token_to_span
    and merge their bboxes into one tight box
  - build an ErrorSpan with that box, the error type, and confidence
"""

import torch
import torch.nn.functional as F

from ocr.tokens import LineSpan
from model.preprocess import Chunk
from model.schemas import ErrorSpan, ID2LABEL, ERROR_TYPES


def build_error_spans(
    chunks: list[Chunk],
    label_id_sequences: list[list[int]],
    spans: list[LineSpan],
    logits: list[list[list[float]]] | None = None,
) -> list[ErrorSpan]:
    """
    chunks             - from preprocess.build_chunks()
    label_id_sequences - from predict.predict(), one list per chunk
    spans              - original list[LineSpan] from ocr pipeline
    logits             - optional raw logits for confidence scores,
                         same shape as label_id_sequences but with num_labels floats per token
                         if None, confidence defaults to 1.0 for all spans
    """
    error_spans = []

    for chunk, label_ids, chunk_logits in _zip_with_logits(chunks, label_id_sequences, logits):
        chunk_errors = _process_chunk(chunk, label_ids, chunk_logits, spans)
        error_spans.extend(chunk_errors)

    return error_spans


def _zip_with_logits(chunks, label_id_sequences, logits):
    if logits is None:
        for chunk, label_ids in zip(chunks, label_id_sequences):
            yield chunk, label_ids, None
    else:
        for chunk, label_ids, chunk_logits in zip(chunks, label_id_sequences, logits):
            yield chunk, label_ids, chunk_logits


def _process_chunk(
    chunk: Chunk,
    label_ids: list[int],
    chunk_logits: list[list[float]] | None,
    spans: list[LineSpan],
) -> list[ErrorSpan]:
    error_spans = []

    # active span being built
    current_type = None       # e.g. "SPELL", "CITE"
    current_span_indices = [] # LineSpan indices contributing to this span
    current_token_probs = []  # per-token probabilities for confidence

    for token_pos, (label_id, span_idx) in enumerate(zip(label_ids, chunk.token_to_span)):

        label = ID2LABEL[label_id]

        # get confidence for this token position if logits were passed
        if chunk_logits is not None:
            token_probs = F.softmax(torch.tensor(chunk_logits[token_pos]), dim=-1)
            pred_prob = token_probs[label_id].item()
        else:
            pred_prob = 1.0

        if label.startswith("B-"):
            # flush any span we were building before starting a new one
            if current_type is not None:
                span = _build_span(current_type, current_span_indices, current_token_probs, spans)
                if span:
                    error_spans.append(span)

            error_type = label[2:]  # strip "B-"
            current_type = error_type
            current_span_indices = [span_idx] if span_idx is not None else []
            current_token_probs = [pred_prob]

        elif label.startswith("I-"):
            error_type = label[2:]  # strip "I-"

            if current_type == error_type:
                # valid continuation of the current span
                if span_idx is not None:
                    current_span_indices.append(span_idx)
                current_token_probs.append(pred_prob)
            else:
                # I- label without a matching B- before it, or type mismatch
                # treat it as a B- to recover gracefully
                if current_type is not None:
                    span = _build_span(current_type, current_span_indices, current_token_probs, spans)
                    if span:
                        error_spans.append(span)
                current_type = error_type
                current_span_indices = [span_idx] if span_idx is not None else []
                current_token_probs = [pred_prob]

        else:
            # O label — flush current span if any
            if current_type is not None:
                span = _build_span(current_type, current_span_indices, current_token_probs, spans)
                if span:
                    error_spans.append(span)
                current_type = None
                current_span_indices = []
                current_token_probs = []

    # flush any span still open at the end of the chunk
    if current_type is not None:
        span = _build_span(current_type, current_span_indices, current_token_probs, spans)
        if span:
            error_spans.append(span)

    return error_spans


def _build_span(
    error_type: str,
    span_indices: list[int],
    token_probs: list[float],
    spans: list[LineSpan],
) -> ErrorSpan | None:
    if not span_indices:
        return None

    # deduplicate span indices while preserving order
    seen = set()
    unique_indices = [i for i in span_indices if not (i in seen or seen.add(i))]

    source_spans = [spans[i] for i in unique_indices if i < len(spans)]
    if not source_spans:
        return None

    # merge bboxes — tight box covering all contributing LineSpans
    x0 = min(s.x0 for s in source_spans)
    y0 = min(s.y0 for s in source_spans)
    x1 = max(s.x1 for s in source_spans)
    y1 = max(s.y1 for s in source_spans)

    # combine text from all contributing spans
    text = " ".join(s.text for s in source_spans)

    # confidence = mean probability across all tokens in this span
    confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0

    return ErrorSpan(
        text=text,
        error_type=ERROR_TYPES.get(error_type, error_type.lower()),
        page_no=source_spans[0].page_no,
        x0=x0, y0=y0, x1=x1, y1=y1,
        confidence=confidence,
    )