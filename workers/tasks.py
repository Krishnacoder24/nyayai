"""
the one Celery task: process_pdf. deliberately thin - all real logic lives
in services/analysis.py so it can be tested and reused without needing a
Celery worker running.
"""

from workers.celery_app import app
from services.analysis import run_analysis


@app.task(name="workers.tasks.process_pdf", bind=True)
def process_pdf(self, job_id: str) -> dict:
    return run_analysis(job_id)