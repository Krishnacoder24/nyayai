"""
end-to-end smoke test: pdf in, errors out - no docker, no celery, no
FastAPI server needed. the fastest way to check "did I break something"
after a change, without spinning up the full stack.

usage:
    uv run python scripts/smoke_test.py some_document.pdf
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/temp/smoke_test"))
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"file not found: {args.pdf}")
        sys.exit(1)

    from ocr.pipeline import extract
    from pipeline.engine import analyze
    from renderer.annotate_pdf import annotate
    from renderer.report import build_report
    from renderer.html_report import render_html

    print(f"extracting: {args.pdf.name}")
    spans = extract(args.pdf)
    print(f"  {len(spans)} lines across {len({s.page_no for s in spans})} pages")

    print("analyzing...")
    errors = analyze(spans)
    print(f"  {len(errors)} error(s) found")

    args.out.mkdir(parents=True, exist_ok=True)

    annotated_path = args.out / f"{args.pdf.stem}_annotated.pdf"
    annotate(args.pdf, errors, annotated_path)
    print(f"annotated PDF: {annotated_path}")

    report = build_report(errors, source_filename=args.pdf.name)
    report_path = args.out / f"{args.pdf.stem}_report.html"
    render_html(report, report_path)
    print(f"HTML report: {report_path}")

    print()
    print("by type:", report["errors_by_type"])

    if not errors:
        print("\nno errors found - if that's unexpected, check model/checkpoint/ has weights and qdrant is running")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
