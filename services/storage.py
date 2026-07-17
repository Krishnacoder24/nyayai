"""
job-id based file layout. one job_id, one uploaded PDF, one annotated PDF,
one JSON report, one HTML report - all named after the same job_id so
nothing has to track a mapping between them separately.

data/uploads/{job_id}.pdf
data/outputs/{job_id}_annotated.pdf
data/outputs/{job_id}_report.json
data/outputs/{job_id}_report.html

no cleanup here - the roadmap already flags "no output cleanup" as a known
gap to fix before anything beyond local single-user use.
"""

from pathlib import Path

from config.settings import settings


def upload_path(job_id: str) -> Path:
    return Path(settings.uploads_dir) / f"{job_id}.pdf"


def annotated_pdf_path(job_id: str) -> Path:
    return Path(settings.outputs_dir) / f"{job_id}_annotated.pdf"


def report_json_path(job_id: str) -> Path:
    return Path(settings.outputs_dir) / f"{job_id}_report.json"


def report_html_path(job_id: str) -> Path:
    return Path(settings.outputs_dir) / f"{job_id}_report.html"


def save_upload(job_id: str, file_bytes: bytes) -> Path:
    path = upload_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(file_bytes)
    return path