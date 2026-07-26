"""
evaluates the trained model against REAL documents with real, hand-
labeled errors - not synthetic corruption. this is separate from
train/evaluate.py on purpose: that script measures how well the model
detects OUR OWN corruption heuristics (swapped prepositions, QWERTY
typos, arithmetic citation offsets), which is a meaningful but
different question from "does this catch errors that actually occur
in a real FIR/contract/court notice". a great synthetic F1 score can
coexist with a model that's just learned to recognize the shape of our
own corruption functions - this script is how you'd actually find that
out, one way or the other.

GROUND TRUTH FORMAT (see ground_truth_schema.md for the full spec):
one JSON file per real document, matched by filename stem to the PDF
it labels (e.g. sample_fir_01.pdf <-> sample_fir_01.json), containing
a list of {"text": <exact substring from the document>, "type": one
of GRAM/CITE/SPELL/ENT, "note": optional human explanation}. matching
on exact text rather than character/token offsets deliberately avoids
needing the ground truth and the model's internal indexing to agree
on tokenization - a human labeler can just copy-paste the offending
phrase straight out of the document.

SCOPE - this evaluates the MODEL specifically, not the full merged
production pipeline: extract -> build_chunks -> predict ->
build_error_spans (model/preprocess.py + model/predict.py +
model/postprocess.py). services/analysis.py's real request path calls
pipeline.engine.analyze(spans) instead, which per this project's own
architecture also merges in the rule-based checkers
(rules/citation_checker.py, rules/entity_checker.py) and dedupes
before returning. that's deliberately NOT what this script tests -
mixing rule-based and ML results together would make it impossible to
tell whether a catch (or a miss) came from the model or from a rule.
if you also want to benchmark the full merged pipeline, that needs
pipeline/engine.py's actual interface, which I don't have here.

NOTE on ENT labels: model/schemas.py's LABELS includes B-ENT/I-ENT
(entity-consistency errors), but scripts/generate_data.py never
generates any ENT training examples - only GRAM/CITE/SPELL. In
practice this means the model has almost certainly never seen a
positive ENT example during training, so real-world ENT predictions
(if the model ever emits any) should be treated as untrained/
essentially noise, not a meaningful signal - ground truth CAN include
"ENT" entries if you want to confirm this empirically, but don't be
surprised if recall on that specific type is near zero regardless of
overall model quality.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

from ocr.pipeline import extract
from model.preprocess import build_chunks
from model.predict import predict
from model.postprocess import build_error_spans

logger = logging.getLogger(__name__)

# ErrorSpan.error_type comes back in long form ("spelling", "grammar",
# "citation", "entity" - see model/schemas.py's ERROR_TYPES), but ground
# truth and DetectedError use the short BIO-prefix form ("SPELL", "GRAM",
# "CITE", "ENT") to match the model's actual LABELS - this is the
# reverse of postprocess.py's own ERROR_TYPES mapping.
_LONG_TO_SHORT_TYPE = {"spelling": "SPELL", "grammar": "GRAM", "citation": "CITE", "entity": "ENT"}


@dataclass
class DetectedError:
    """one error span, in the same shape whether it came from the real
    model or from a hand-labeled ground truth file - matching happens
    on (text, type) pairs, not positions."""
    text: str
    error_type: str  # "GRAM" | "CITE" | "SPELL"
    note: Optional[str] = None


def load_ground_truth(json_path: Path) -> List[DetectedError]:
    with open(json_path) as f:
        data = json.load(f)
    return [
        DetectedError(text=e["text"], error_type=e["type"], note=e.get("note"))
        for e in data
    ]


def _run_pipeline_on_document(pdf_path: Path) -> List[DetectedError]:
    """runs the model-only path (not the full merged production
    pipeline - see this module's docstring for why): OCR extraction,
    chunking, inference (with real logits so build_error_spans computes
    genuine per-span confidence instead of its 1.0 fallback), then
    reconstructing ErrorSpans with real bboxes."""
    spans = extract(pdf_path)
    chunks = build_chunks(spans)
    label_id_sequences, logits = predict(chunks, return_logits=True)
    error_spans = build_error_spans(chunks, label_id_sequences, spans, logits=logits)

    return [
        DetectedError(
            text=e.text,
            error_type=_LONG_TO_SHORT_TYPE.get(e.error_type, e.error_type.upper()),
            note=f"confidence={e.confidence:.2f} page={e.page_no}",
        )
        for e in error_spans
    ]


def score(predicted: List[DetectedError], ground_truth: List[DetectedError]) -> Dict:
    """precision/recall/F1 at the span level, matching on (text, type).
    a prediction counts as a true positive if the SAME text (case-
    insensitive, whitespace-normalized) was labeled with the SAME error
    type in ground truth - deliberately strict on type (a real CITE
    error flagged as GRAM doesn't count as a hit, since the wrong
    error_type would mislead a user about what's actually wrong)."""

    def _norm(e: DetectedError) -> tuple:
        return (" ".join(e.text.lower().split()), e.error_type)

    pred_set = {_norm(e) for e in predicted}
    gold_set = {_norm(e) for e in ground_truth}

    true_positives = pred_set & gold_set
    false_positives = pred_set - gold_set
    false_negatives = gold_set - pred_set

    tp, fp, fn = len(true_positives), len(false_positives), len(false_negatives)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "missed_errors": [dict(zip(("text", "type"), e)) for e in false_negatives],
        "false_alarms": [dict(zip(("text", "type"), e)) for e in false_positives],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=Path, required=True,
                        help="directory containing real PDFs + matching ground-truth .json files "
                             "(same filename stem, e.g. sample_01.pdf + sample_01.json)")
    args = parser.parse_args()

    pdfs = sorted(args.documents.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {args.documents}")
        return

    all_predicted, all_gold = [], []
    per_document_results = []

    for pdf_path in pdfs:
        gt_path = pdf_path.with_suffix(".json")
        if not gt_path.exists():
            print(f"  skipping {pdf_path.name}: no matching ground truth file ({gt_path.name})")
            continue

        ground_truth = load_ground_truth(gt_path)
        try:
            predicted = _run_pipeline_on_document(pdf_path)
        except Exception as e:
            print(f"  ERROR running pipeline on {pdf_path.name}: {e}")
            continue

        result = score(predicted, ground_truth)
        per_document_results.append({"document": pdf_path.name, **result})
        all_predicted.extend(predicted)
        all_gold.extend(ground_truth)

    if not per_document_results:
        print("\nNo documents were actually evaluated - check for errors above, "
              "or confirm ground truth .json files exist alongside the PDFs.")
        return

    overall = score(all_predicted, all_gold)
    print("\n=== Per-document results ===")
    for r in per_document_results:
        print(f"  {r['document']}: P={r['precision']} R={r['recall']} F1={r['f1']} "
              f"(TP={r['true_positives']} FP={r['false_positives']} FN={r['false_negatives']})")

    print("\n=== Overall (real-document benchmark) ===")
    print(f"  Precision: {overall['precision']}")
    print(f"  Recall:    {overall['recall']}")
    print(f"  F1:        {overall['f1']}")
    if overall["missed_errors"]:
        print(f"\n  Missed errors ({len(overall['missed_errors'])}):")
        for m in overall["missed_errors"][:10]:
            print(f"    - [{m['type']}] {m['text']}")
    if overall["false_alarms"]:
        print(f"\n  False alarms ({len(overall['false_alarms'])}):")
        for m in overall["false_alarms"][:10]:
            print(f"    - [{m['type']}] {m['text']}")


if __name__ == "__main__":
    main()