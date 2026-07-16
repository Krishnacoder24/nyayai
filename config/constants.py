# -------------------------
# InLegalBERT
# -------------------------

MAX_TOKENS = 512

CHUNK_STRIDE = 128

INFERENCE_BATCH_SIZE = 8

# -------------------------
# Rendering
# -------------------------

ERROR_COLORS = {
    "spelling": "#FFD700",
    "grammar": "#FFA500",
    "citation": "#FF4444",
    "entity": "#00BFFF",
}

# Embeddings constants for corpus/embeddings.py
MODEL_NAME = "law-ai/InLegalBERT"
BATCH_SIZE = 16

# acts for parsing and chunking
ACTS = ["ipc", "bns", "bnss", "cpc", "constitution"]

# Rendering constants for anotate_pdf.py, report.py, html_report.py, colors.py
FILL_ALPHA = 0.35
STROKE_ALPHA = 0.9
STROKE_WIDTH = 1.2