"""
queue names + routing. one queue for now - if OCR-heavy jobs ever need to
be separated from lighter ones (e.g. so a big scanned FIR doesn't block a
quick native-PDF contract check), split into two queues here and update
task_routes without touching tasks.py or api/.
"""

PDF_PROCESSING_QUEUE = "pdf_processing"

TASK_ROUTES = {
    "workers.tasks.process_pdf": {"queue": PDF_PROCESSING_QUEUE},
}