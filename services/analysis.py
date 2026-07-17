"""
orchestrates one document through the full pipeline:
  ocr.pipeline.extract -> pipeline.engine.analyze -> renderer -> save

this is the only file workers/tasks.py calls into - Celery knows nothing
about OCR, the model, or rendering, it just calls run_analysis(job_id).
"""

from ocr.pipeline import extract
from pipeline.engine import analyze
from renderer.annotate_pdf import annotate
from renderer.report import build_report
from renderer.html_report import render_html


from services.storage import (
    upload_path,
    annotated_pdf_path,
    report_json_path,
    report_html_path,
)

import json


def run_analysis(job_id: str) -> dict:
    """
    runs the full pipeline for an already-uploaded PDF (see
    services.storage.save_upload) and writes every output file.
    returns the report dict - this becomes the Celery task's result,
    in addition to being saved as report_json_path(job_id).
    """
    pdf_path = upload_path(job_id)
    source_filename = pdf_path.name

    spans = extract(pdf_path)
    errors = analyze(spans)

    annotate(pdf_path, errors, annotated_pdf_path(job_id))

    report = build_report(errors, source_filename=source_filename)
    report_json_path(job_id).parent.mkdir(parents=True, exist_ok=True)
    report_json_path(job_id).write_text(json.dumps(report, indent=2))

    render_html(report, report_html_path(job_id))

    return report # returning the report dict allows the Celery task to return it as its result, which can be useful for logging, debugging, or further processing.