import { useCallback, useMemo, useState } from 'react'
import './App.css'
import UploadPage from './UploadPage'
import PdfCanvas from './PdfCanvas'
import HighlightOverlay from './HighlightOverlay'
import MarginRail from './MarginRail'
import ErrorList from './ErrorList'
import { uploadPdf, pollJobStatus, fetchResult } from './api'

const POLL_INTERVAL_MS = 300

export default function App() {
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState(null)
  const [report, setReport] = useState(null)
  const [uploadError, setUploadError] = useState(null)
  const [currentPage, setCurrentPage] = useState(1)
  const [activeErrorIndex, setActiveErrorIndex] = useState(null)
  const [pageInfo, setPageInfo] = useState(null) // { widthPts, heightPts, displayScale }

  const handleFileSelected = useCallback(async (selectedFile) => {
    setFile(selectedFile)
    setStatus('PENDING')
    setReport(null)
    setUploadError(null)

    let jobId
    try {
      const uploadResult = await uploadPdf(selectedFile)
      jobId = uploadResult.jobId
    } catch (err) {
      setStatus(null)
      setUploadError(err.message)
      return
    }

    const poll = setInterval(async () => {
      try {
        const { status: jobStatus } = await pollJobStatus(jobId)
        setStatus(jobStatus)

        if (jobStatus === 'SUCCESS') {
          clearInterval(poll)
          const result = await fetchResult(jobId)
          setReport(result)
        } else if (jobStatus === 'FAILURE') {
          clearInterval(poll)
          // fetchResult throws with the real error message on a failed job -
          // this is how we surface *why* it failed, not just that it did
          try {
            await fetchResult(jobId)
          } catch (err) {
            setStatus(null)
            setUploadError(err.message)
          }
        }
      } catch (err) {
        clearInterval(poll)
        setStatus(null)
        setUploadError(err.message)
      }
    }, POLL_INTERVAL_MS)
  }, [])

  const pageErrors = useMemo(
    () => report?.errors.filter((e) => e.page_no === currentPage) ?? [],
    [report, currentPage],
  )

  function selectError(globalIndex) {
    const error = report.errors[globalIndex]
    setActiveErrorIndex(globalIndex)
    if (error.page_no !== currentPage) setCurrentPage(error.page_no)
  }

  function selectPageError(pageLocalIndex) {
    const error = pageErrors[pageLocalIndex]
    const globalIndex = report.errors.indexOf(error)
    setActiveErrorIndex(globalIndex)
  }

  const activeIndexOnPage = useMemo(() => {
    if (activeErrorIndex === null || !report) return null
    const active = report.errors[activeErrorIndex]
    if (active.page_no !== currentPage) return null
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
        <div className="viewer-page-nav">
          <button
            type="button"
            disabled={currentPage <= 1}
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
          >
            ← Prev
          </button>
          <span className="viewer-page-label">Page {currentPage}</span>
          <button type="button" onClick={() => setCurrentPage((p) => p + 1)}>
            Next →
          </button>
        </div>
      </header>

      <div className="viewer-body">
        <ErrorList
          report={report}
          activeErrorIndex={activeErrorIndex}
          currentPage={currentPage}
          onSelect={selectError}
        />

        <div className="viewer-canvas-area">
          <div className="viewer-canvas-stack">
            <PdfCanvas file={file} pageNumber={currentPage} onPageRendered={setPageInfo} />
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