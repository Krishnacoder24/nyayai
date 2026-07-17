/*
  upload / poll / fetch result - matches the real backend flow:
  POST /upload -> {job_id}, GET /status/{job_id} -> {status}, GET /result/{job_id} -> report.
*/


const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export async function uploadPdf(file) {
  const formData = new FormData()
  formData.append('file', file)

  const res = await fetch(`${API_BASE_URL}/upload`, { method: 'POST', body: formData })
  const data = await res.json().catch(() => ({}))

  if (!res.ok) {
    throw new Error(data.detail || `upload failed (${res.status})`)
  }

  return { jobId: data.job_id }
}

export async function pollJobStatus(jobId) {
  const res = await fetch(`${API_BASE_URL}/status/${jobId}`)
  const data = await res.json().catch(() => ({}))

  if (!res.ok) {
    throw new Error(data.detail || `status check failed (${res.status})`)
  }

  return { status: data.status }
}

export async function fetchResult(jobId) {
  const res = await fetch(`${API_BASE_URL}/result/${jobId}`)
  const data = await res.json().catch(() => ({}))

  if (!res.ok) {
    throw new Error(data.detail || `fetching result failed (${res.status})`)
  }

  if (data.status === 'FAILURE') {
    throw new Error(data.error || 'processing failed')
  }

  // report is shaped exactly like renderer/report.py's build_report() output -
  // same shape mockData.js used to fake, so ErrorList/PdfCanvas/HighlightOverlay/
  // MarginRail don't need to know anything changed. the two download URLs are
  // folded in as extra fields, made absolute against the API origin.
  return {
    ...data.report,
    annotated_pdf_url: data.annotated_pdf_url ? `${API_BASE_URL}${data.annotated_pdf_url}` : null,
    report_html_url: data.report_html_url ? `${API_BASE_URL}${data.report_html_url}` : null,
  }
}




