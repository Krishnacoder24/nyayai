"""
POST /upload - accepts a PDF, saves it under its own job_id, enqueues
process_pdf, returns the job_id immediately. everything after this is
polled via GET /status/{job_id} and GET /result/{job_id}.
"""

import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException

from api.schemas.upload import UploadResponse
from config import constants
from services.storage import save_upload
from workers.tasks import process_pdf

router = APIRouter()

MAX_UPLOAD_BYTES = constants.MAX_UPLOAD_BYTES  # 50MB - generous for a scanned multi-page FIR


@router.post("/upload", response_model=UploadResponse) #@router means this function is a route handler for the /upload endpoint, and it will return a response model of type UploadResponse.
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="only PDF files are accepted")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file too large (max 50MB)")
    if not file_bytes:
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    job_id = str(uuid.uuid4())
    # file.filename is whatever the client's browser sent (e.g.
    # "FIR_2024.pdf") - saved alongside the upload purely so the report
    # can show the real filename later instead of the internal job_id.pdf
    # storage name. never used to build a filesystem path anywhere - every
    # path in services/storage.py is keyed on job_id, not this string.
    save_upload(job_id, file_bytes, file.filename)

    # IMPORTANT: task_id must be forced to equal job_id here, or GET
    # /status/{job_id} and /result/{job_id} (jobs.py) - which both call
    # AsyncResult(job_id, ...) - end up asking Celery about a task ID
    # that was never actually used. .delay(job_id) passes job_id as
    # process_pdf's FUNCTION ARGUMENT only; Celery auto-generates its
    # own separate random UUID as the real task ID unless task_id is
    # explicitly passed to apply_async(). .delay(*args, **kwargs) is
    # just shorthand for apply_async(args=args, kwargs=kwargs) -
    # switching to apply_async directly here is the minimal change
    # needed to also pass task_id.
    process_pdf.apply_async(args=[job_id], task_id=job_id)
  

    return UploadResponse(job_id=job_id)