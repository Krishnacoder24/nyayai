import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import UploadPage from './UploadPage'
import PdfCanvas from './PdfCanvas'
import HighlightOverlay from './HighlightOverlay'
import MarginRail from './MarginRail'
import ErrorList from './ErrorList'
import { uploadPdf, pollJobStatus, fetchResult } from './api'

const POLL_INTERVAL_MS = 300
const ZOOM_STEP = 0.15
const ZOOM_MIN = 0.4
const ZOOM_MAX = 3

export default function App() {
  const [file, setFile] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState(null)
  const [report, setReport] = useState(null)
  const [uploadError, setUploadError] = useState(null)
  const [currentPage, setCurrentPage] = useState(1)
  const [activeErrorIndex, setActiveErrorIndex] = useState(null)
  const [pageInfo, setPageInfo] = useState(null) // { widthPts, heightPts, displayScale, numPages }
  const [zoom, setZoom] = useState(1) // user multiplier on top of fit-to-width; 1 = fit exactly

  // PdfCanvas sizes itself to fill this rather than a fixed 1-point-per-
  // CSS-pixel size (which renders as a small rectangle in a sea of empty
  // space on any normal monitor - see PdfCanvas.jsx's docstring). tracked
  // via ResizeObserver rather than window resize alone, since the sidebar
  // or margin rail resizing (not just the window) also changes how much
  // width is actually available.
  const canvasAreaRef = useRef(null)
  const [containerWidth, setContainerWidth] = useState(null)

  useEffect(() => {
    const el = canvasAreaRef.current
    if (!el) return

    // leaves room for the canvas area's own horizontal padding (2rem each
    // side, see .viewer-canvas-area) and the margin rail beside it, so
    // "fit width" doesn't fit right up against - or past - the scrollbar.
    const HORIZONTAL_ALLOWANCE = 96

    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width
      if (width) setContainerWidth(Math.max(200, width - HORIZONTAL_ALLOWANCE))
    })
    observer.observe(el)
    return () => observer.disconnect()
    // deps: [report], not [] - canvasAreaRef.current is null until the
    // viewer JSX (below, behind `if (!report) return <UploadPage/>`)
    // actually mounts, which only happens once report is first set. an
    // empty dep array would run this exactly once, before that div
    // exists, capture a null ref, and never observe anything.
  }, [report])

  const zoomIn = useCallback(() => setZoom((z) => Math.min(ZOOM_MAX, +(z + ZOOM_STEP).toFixed(2))), [])
  const zoomOut = useCallback(() => setZoom((z) => Math.max(ZOOM_MIN, +(z - ZOOM_STEP).toFixed(2))), [])
  const resetZoom = useCallback(() => setZoom(1), [])

  const handleFileSelected = useCallback(async (selectedFile) => {
    setFile(selectedFile)
    setJobId(null)
    setStatus('PENDING')
    setReport(null)
    setUploadError(null)

    try {
      const uploadResult = await uploadPdf(selectedFile)
      setJobId(uploadResult.jobId)
    } catch (err) {
      setStatus(null)
      setUploadError(err.message)
    }
  }, [])

  // poll job status once we have a jobId. this lives in its own effect,
  // keyed on jobId, specifically so React can clean it up: on unmount, or
  // if a new upload starts and jobId changes, the returned cleanup clears
  // the interval and flips `cancelled` so any already-in-flight
  // pollJobStatus/fetchResult response is ignored instead of calling
  // setState after the fact. previously this setInterval was created
  // inside the same plain async callback that did the upload, with no
  // cleanup path at all - navigating away mid-poll left the interval
  // running forever, still calling setStatus/setReport against state that
  // no one was reading anymore.
  useEffect(() => {
    if (!jobId) return
    let cancelled = false

    const poll = setInterval(async () => {
      try {
        const { status: jobStatus } = await pollJobStatus(jobId)
        if (cancelled) return
        setStatus(jobStatus)

        if (jobStatus === 'SUCCESS') {
          clearInterval(poll)
          const result = await fetchResult(jobId)
          if (!cancelled) setReport(result)
        } else if (jobStatus === 'FAILURE') {
          clearInterval(poll)
          // fetchResult throws with the real error message on a failed job -
          // this is how we surface *why* it failed, not just that it did
          try {
            await fetchResult(jobId)
          } catch (err) {
            if (!cancelled) {
              setStatus(null)
              setUploadError(err.message)
            }
          }
        }
      } catch (err) {
        clearInterval(poll)
        if (!cancelled) {
          setStatus(null)
          setUploadError(err.message)
        }
      }
    }, POLL_INTERVAL_MS)

    return () => {
      cancelled = true
      clearInterval(poll)
    }
  }, [jobId])

  // report.total_pages comes straight from renderer/report.py and is known
  // the instant the report loads - pageInfo.numPages only exists after
  // pdf.js finishes parsing the file client-side, a moment later. prefer
  // the backend value; fall back to pdf.js's count if a report predates
  // this field (e.g. one saved before total_pages was added).
  const totalPages = report?.total_pages || pageInfo?.numPages || null

  // report.errors[].page_no is 0-indexed (see ocr/native_extractor.py and
  // ocr/surya_extractor.py, both built via plain enumerate(pages)/pymupdf's
  // doc[i]) - currentPage is 1-indexed to match pdf.js's getPage() and the
  // "Page 1" label shown in the header, so every read/write of page_no
  // needs a +/-1 conversion at this boundary. getting this wrong doesn't
  // error out, it just silently shows zero highlights on the very page
  // an error is actually on.
  const pageErrors = useMemo(
    () => report?.errors.filter((e) => e.page_no === currentPage - 1) ?? [],
    [report, currentPage],
  )

  function selectError(globalIndex) {
    const error = report.errors[globalIndex]
    setActiveErrorIndex(globalIndex)
    if (error.page_no !== currentPage - 1) setCurrentPage(error.page_no + 1)
  }

  function selectPageError(pageLocalIndex) {
    const error = pageErrors[pageLocalIndex]
    const globalIndex = report.errors.indexOf(error)
    setActiveErrorIndex(globalIndex)
  }

  const activeIndexOnPage = useMemo(() => {
    if (activeErrorIndex === null || !report) return null
    const active = report.errors[activeErrorIndex]
    if (active.page_no !== currentPage - 1) return null
    return pageErrors.indexOf(active)
  }, [activeErrorIndex, report, currentPage, pageErrors])

  if (!report) {
    return <UploadPage onFileSelected={handleFileSelected} status={status} error={uploadError} />
  }

  return (
    <div className="viewer">
      <header className="viewer-header">
        <div className="viewer-header-title">
          <span className="viewer-eyebrow">NyayAI</span>
          <h1>{report.source_filename}</h1>
        </div>
        <div className="viewer-header-controls">
          <div className="viewer-page-nav">
            <button
              type="button"
              disabled={currentPage <= 1}
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            >
              ← Prev
            </button>
            <span className="viewer-page-label">Page {currentPage}</span>
            <button
              type="button"
              disabled={totalPages != null && currentPage >= totalPages}
              onClick={() => setCurrentPage((p) => Math.min(totalPages ?? p + 1, p + 1))}
            >
              Next →
            </button>
          </div>

          <div className="viewer-zoom-nav">
            <button type="button" onClick={zoomOut} disabled={zoom <= ZOOM_MIN} aria-label="Zoom out">
              −
            </button>
            <button type="button" className="viewer-zoom-reset" onClick={resetZoom} title="Reset to fit width">
              {Math.round(zoom * 100)}%
            </button>
            <button type="button" onClick={zoomIn} disabled={zoom >= ZOOM_MAX} aria-label="Zoom in">
              +
            </button>
          </div>
        </div>
      </header>

      <div className="viewer-body">
        <ErrorList
          report={report}
          activeErrorIndex={activeErrorIndex}
          onSelect={selectError}
        />

        <div className="viewer-canvas-area" ref={canvasAreaRef}>
          <div className="viewer-canvas-stack">
            <PdfCanvas
              file={file}
              pageNumber={currentPage}
              containerWidth={containerWidth}
              zoom={zoom}
              onPageRendered={setPageInfo}
            />
            <HighlightOverlay
              errors={pageErrors}
              displayScale={pageInfo?.displayScale}
              activeErrorIndex={activeIndexOnPage}
              onSelect={selectPageError}
            />
          </div>
          <MarginRail
            errors={pageErrors}
            pageHeightPts={pageInfo?.heightPts}
            activeErrorIndex={activeIndexOnPage}
            onSelect={selectPageError}
          />
        </div>
      </div>
    </div>
  )
}