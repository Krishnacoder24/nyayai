"""API response schemas."""
from typing import Literal, Optional
from pydantic import BaseModel

# mirrors Celery's own states - see the job-status-tracking discussion:
# this is the "rely on Celery's result backend directly" approach, so the
# states the frontend sees are exactly Celery's, not a custom state machine
JobState = Literal["PENDING", "STARTED", "SUCCESS", "FAILURE", "RETRY", "REVOKED"]


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobState


class JobResultResponse(BaseModel):
    job_id: str
    status: JobState
    report: Optional[dict] = None
    annotated_pdf_url: Optional[str] = None
    report_html_url: Optional[str] = None
    error: Optional[str] = None