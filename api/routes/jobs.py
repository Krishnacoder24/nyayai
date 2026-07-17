"""
GET /status/{job_id} - polls Celery's own result backend directly (no
custom state machine - see the job-status-tracking discussion). states
are exactly Celery's: PENDING, STARTED, SUCCESS, FAILURE, RETRY, REVOKED.

GET /result/{job_id} - once SUCCESS, returns the report plus download
links for the annotated PDF and HTML report.
"""

from fastapi import APIRouter, HTTPException
from celery.result import AsyncResult

from api.schemas.response import JobStatusResponse, JobResultResponse
from services.storage import annotated_pdf_path, report_html_path
from workers.celery_app import app as celery_app

router = APIRouter()


@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_status(job_id: str):
    result = AsyncResult(job_id, app=celery_app)
    return JobStatusResponse(job_id=job_id, status=result.status)


@router.get("/result/{job_id}", response_model=JobResultResponse)
def get_result(job_id: str):
    result = AsyncResult(job_id, app=celery_app)

    if result.status == "FAILURE":
        return JobResultResponse(job_id=job_id, status=result.status, error=str(result.result))

    if result.status != "SUCCESS":
        raise HTTPException(status_code=409, detail=f"job is not finished yet (status: {result.status})")

    report = result.result

    annotated_url = None
    if annotated_pdf_path(job_id).exists():
        annotated_url = f"/files/{job_id}_annotated.pdf"

    html_url = None
    if report_html_path(job_id).exists():
        html_url = f"/files/{job_id}_report.html"

    return JobResultResponse(
        job_id=job_id,
        status=result.status,
        report=report,
        annotated_pdf_url=annotated_url,
        report_html_url=html_url,
    )