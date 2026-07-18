"""
speed + accuracy benchmarks.

speed: OCR extraction time (per page) and, if a fine-tuned checkpoint
exists, model inference time (per chunk).

accuracy: if a checkpoint AND data/training/test.jsonl both exist, reuses
train/evaluate.py's exact scoring path - so these numbers always match
what `uv run python -m train.evaluate` reports directly, rather than a
second implementation that could quietly drift out of sync with it. a
checkpoint that's fast but scores near-zero F1 isn't actually useful, so
speed and accuracy are reported together rather than as separate tools.

usage:
    uv run python scripts/benchmark.py --pdf some_document.pdf
"""

import argparse
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, help="a PDF to benchmark OCR speed against")
    args = parser.parse_args()

    print("=" * 60)
    print("NyayAI benchmark")
    print("=" * 60)

    if args.pdf:
        benchmark_ocr(args.pdf)

    benchmark_model_inference()
    benchmark_accuracy()


def benchmark_ocr(pdf_path: Path):
    from ocr.pipeline import extract

    print(f"\n--- OCR speed: {pdf_path.name} ---")
    start = time.monotonic()
    spans = extract(pdf_path)
    elapsed = time.monotonic() - start

    pages = len({s.page_no for s in spans}) if spans else 0
    per_page = elapsed / pages if pages else 0

    print(f"pages: {pages}")
    print(f"lines extracted: {len(spans)}")
    print(f"total time: {elapsed:.2f}s")
    print(f"per page: {per_page:.2f}s/page")


def benchmark_model_inference():
    from model.predict import _checkpoint_exists

    print("\n--- model inference speed ---")
    if not _checkpoint_exists():
        print("no fine-tuned checkpoint in model/checkpoint/ - skipping (predict.py would return all-O anyway)")
        return

    from model.preprocess import build_chunks
    from model.predict import predict
    from ocr.tokens import LineSpan

    # synthetic spans just to exercise the model at a known size - this
    # measures raw inference throughput, not any particular document
    sample_text = "whoever commits murder shall be punished with death or imprisonment for life"
    sample_spans = [LineSpan(text=sample_text, page_no=1, x0=0, y0=0, x1=0, y1=0) for _ in range(50)]
    chunks = build_chunks(sample_spans)

    start = time.monotonic()
    predict(chunks)
    elapsed = time.monotonic() - start

    print(f"chunks: {len(chunks)}")
    print(f"total time: {elapsed:.2f}s")
    if chunks:
        print(f"per chunk: {elapsed / len(chunks):.3f}s")


def benchmark_accuracy():
    from model.predict import _checkpoint_exists

    print("\n--- accuracy on held-out test set ---")
    test_path = Path("data/training/test.jsonl")

    if not _checkpoint_exists():
        print("no fine-tuned checkpoint - skipping")
        return
    if not test_path.exists():
        print(f"{test_path} not found - run scripts/generate_data.py first")
        return

    from train.evaluate import main as run_evaluate
    run_evaluate()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
