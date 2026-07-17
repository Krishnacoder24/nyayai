"""
FastAPI app entrypoint. wires the three route modules together, serves
outputs/ as static files (so the frontend can download the annotated PDF
and HTML report directly), and allows the Vite dev server's origin so
local frontend development isn't blocked by CORS.

no auth - matches the roadmap's own known-limitations list ("no
authentication on the API... must be added before any deployment").
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config.settings import settings
from api.routes import upload, jobs, health

app = FastAPI(title="NyayAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server default
    allow_methods=["*"], # * allows all HTTP methods (GET, POST, PUT, DELETE, etc.) to be used in cross-origin requests. This is useful for development and testing purposes, but should be restricted in production for security reasons.
    allow_headers=["*"], 
)

app.include_router(upload.router)
app.include_router(jobs.router)
app.include_router(health.router)

app.mount("/files", StaticFiles(directory=settings.outputs_dir), name="files") # /files is the URL path where the static files will be served from. StaticFiles is a FastAPI class that serves static files from a specified directory. The directory parameter specifies the local directory where the static files are located, which is set to settings.outputs_dir. The name parameter assigns a name to this static files route, which can be used for reverse URL lookups within the application. So a file saved as data/outputs/<job_id>_annotated.pdf is reachable as /files/<job_id>_annotated.pdf.

