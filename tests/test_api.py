"""
tests for the FastAPI layer: upload flow, status polling, result
fetching, and the health check.

process_pdf.apply_async and AsyncResult are both mocked - a real Celery
worker isn't running in tests (see workers/celery_app.py's own docstring:
the filesystem broker just writes a file, nothing consumes it without a
worker process), so "status" and "result" behaviour is driven by
controlling what AsyncResult reports, exactly the way jobs.py itself
reads it.
"""

import io

import pytest
from fastapi.testclient import TestClient

from config.settings import settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    # isolate every uploaded/generated file under this test's own tmp_path
    # rather than the real data/uploads, data/outputs directories
    monkeypatch.setattr(settings, "uploads_dir", tmp_path / "uploads")
    monkeypatch.setattr(settings, "outputs_dir", tmp_path / "outputs")
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)

    import api.main as main

    return TestClient(main.app)


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------
def test_upload_rejects_non_pdf_content_type(client):
    response = client.post(
        "/upload",
        files={"file": ("notice.txt", io.BytesIO(b"hello"), "text/plain")},
    )

    assert response.status_code == 400
    assert "pdf" in response.json()["detail"].lower()


def test_upload_rejects_empty_file(client):
    response = client.post(
        "/upload",
        files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_rejects_file_over_size_limit(client, monkeypatch):
    import api.routes.upload as upload_route

    monkeypatch.setattr(upload_route, "MAX_UPLOAD_BYTES", 10)

    response = client.post(
        "/upload",
        files={"file": ("big.pdf", io.BytesIO(b"%PDF-" + b"0" * 20), "application/pdf")},
    )

    assert response.status_code == 413


def test_upload_success_saves_file_and_enqueues_job(client, monkeypatch, sample_pdf_bytes):
    import api.routes.upload as upload_route

    captured = {}

    def _fake_apply_async(args, task_id):
        captured["args"] = args
        captured["task_id"] = task_id

    monkeypatch.setattr(upload_route.process_pdf, "apply_async", _fake_apply_async)

    response = client.post(
        "/upload",
        files={"file": ("FIR_2024.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]

    # job_id must be both the file's storage key AND the Celery task_id -
    # see upload.py's own comment on why apply_async(task_id=...) matters:
    # /status and /result both look the task up by job_id.
    assert captured["args"] == [job_id]
    assert captured["task_id"] == job_id
    assert (settings.uploads_dir / f"{job_id}.pdf").exists()
    assert (settings.uploads_dir / f"{job_id}.pdf").read_bytes() == sample_pdf_bytes


# ---------------------------------------------------------------------------
# GET /status/{job_id}
# ---------------------------------------------------------------------------
def test_status_reports_celery_state_verbatim(client, monkeypatch):
    import api.routes.jobs as jobs_route

    class _FakeResult:
        def __init__(self, job_id, app):
            self.status = "STARTED"

    monkeypatch.setattr(jobs_route, "AsyncResult", _FakeResult)

    response = client.get("/status/some-job-id")

    assert response.status_code == 200
    assert response.json() == {"job_id": "some-job-id", "status": "STARTED"}


# ---------------------------------------------------------------------------
# GET /result/{job_id}
# ---------------------------------------------------------------------------
def test_result_returns_409_while_job_is_pending(client, monkeypatch):
    import api.routes.jobs as jobs_route

    class _FakeResult:
        def __init__(self, job_id, app):
            self.status = "PENDING"

    monkeypatch.setattr(jobs_route, "AsyncResult", _FakeResult)

    response = client.get("/result/some-job-id")

    assert response.status_code == 409


def test_result_returns_error_message_on_failure(client, monkeypatch):
    import api.routes.jobs as jobs_route

    class _FakeResult:
        def __init__(self, job_id, app):
            self.status = "FAILURE"
            self.result = RuntimeError("OCR extraction failed")

    monkeypatch.setattr(jobs_route, "AsyncResult", _FakeResult)

    response = client.get("/result/some-job-id")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "FAILURE"
    assert "OCR extraction failed" in body["error"]


def test_result_returns_report_and_file_urls_on_success(client, monkeypatch):
    import api.routes.jobs as jobs_route

    job_id = "success-job-id"
    report = {
        "source_filename": "FIR_2024.pdf",
        "total_errors": 1,
        "errors_by_type": {"citation": 1},
        "errors": [{"text": "Section 999 IPC", "error_type": "citation"}],
    }

    class _FakeResult:
        def __init__(self, jid, app):
            self.status = "SUCCESS"
            self.result = report

    monkeypatch.setattr(jobs_route, "AsyncResult", _FakeResult)

    # simulate the annotated PDF + HTML report having actually been
    # written by the (mocked-away) Celery task
    (settings.outputs_dir / f"{job_id}_annotated.pdf").write_bytes(b"%PDF-fake")
    (settings.outputs_dir / f"{job_id}_report.html").write_text("<html></html>")

    response = client.get(f"/result/{job_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUCCESS"
    assert body["report"] == report
    assert body["annotated_pdf_url"] == f"/files/{job_id}_annotated.pdf"
    assert body["report_html_url"] == f"/files/{job_id}_report.html"


def test_result_omits_file_urls_when_outputs_dont_exist_yet(client, monkeypatch):
    import api.routes.jobs as jobs_route

    class _FakeResult:
        def __init__(self, job_id, app):
            self.status = "SUCCESS"
            self.result = {"total_errors": 0}

    monkeypatch.setattr(jobs_route, "AsyncResult", _FakeResult)

    response = client.get("/result/no-files-yet")

    body = response.json()
    assert body["annotated_pdf_url"] is None
    assert body["report_html_url"] is None


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
def test_health_reports_qdrant_reachable(client, monkeypatch):
    import api.routes.health as health_route

    class _FakeClient:
        def __init__(self, url):
            pass

        def get_collections(self):
            return []

    monkeypatch.setattr(health_route, "QdrantClient", _FakeClient)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "qdrant": "reachable"}


def test_health_reports_qdrant_unreachable(client, monkeypatch):
    import api.routes.health as health_route

    class _FakeClient:
        def __init__(self, url):
            pass

        def get_collections(self):
            raise ConnectionError("no route to host")

    monkeypatch.setattr(health_route, "QdrantClient", _FakeClient)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "qdrant": "unreachable"}