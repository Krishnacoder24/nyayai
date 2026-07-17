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
    save_upload(job_id, file_bytes)

    process_pdf.delay(job_id)

    return UploadResponse(job_id=job_id)