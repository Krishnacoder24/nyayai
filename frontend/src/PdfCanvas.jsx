import { useEffect, useRef, useState } from 'react'
import * as pdfjsLib from 'pdfjs-dist'
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.mjs?url'

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl

// render at a fixed internal scale for crispness, independent of CSS display size
const RENDER_SCALE = 1.5

/*
  renders `pageNumber` of `file` onto a canvas.

  reports back { widthPts, heightPts, displayScale, numPages } via
  onPageRendered so HighlightOverlay/MarginRail can convert ErrorSpan
  bboxes (in PDF points, top-left origin - see utils/bbox.py) into
  on-screen pixels, and App.jsx can disable "Next" on the last page. no
  y-flip is needed here: pdf.js's viewport is already top-left-origin,
  same as pdfplumber - confirmed against a real page before writing this
  (unlike renderer/annotate_pdf.py, which DOES need a flip, because
  reportlab's canvas is bottom-left-origin).

  loading and rendering are split into two effects deliberately: loading
  parses the whole PDF from bytes, which is the expensive part, and only
  needs to happen once per `file`. paging back and forth used to redo
  that full parse on every single page turn (`pdfjsLib.getDocument(...)`
  was inside the same effect keyed on `[file, pageNumber]`) - splitting it
  out means turning pages only ever calls the cheap `doc.getPage(...)`.
*/
export default function PdfCanvas({ file, pageNumber, onPageRendered }) {
  const canvasRef = useRef(null)
  const [error, setError] = useState(null)
  const [doc, setDoc] = useState(null)

  useEffect(() => {
    if (!file) {
      setDoc(null)
      return
    }
    let cancelled = false
    let loadedDoc = null

    async function load() {
      try {
        const data = new Uint8Array(await file.arrayBuffer())
        loadedDoc = await pdfjsLib.getDocument({ data }).promise
        if (cancelled) {
          loadedDoc.destroy()
          return
        }
        setDoc(loadedDoc)
      } catch (err) {
        if (!cancelled) setError(err.message)
      }
    }

    load()
    return () => {
      cancelled = true
      // destroy the loaded document's worker-side resources once it's no
      // longer the one being shown - either a new file replaced it, or
      // this component unmounted. without this, every new upload leaked
      // the previous PDF's parsed state in the pdf.js worker.
      loadedDoc?.destroy()
    }
  }, [file])

  useEffect(() => {
    if (!doc) return
    let cancelled = false
    let renderTask = null

    async function render() {
      try {
        const page = await doc.getPage(pageNumber)
        if (cancelled) return

        const viewportAtScale1 = page.getViewport({ scale: 1 })
        const viewport = page.getViewport({ scale: RENDER_SCALE })

        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        canvas.width = viewport.width
        canvas.height = viewport.height
        canvas.style.width = `${viewport.width / RENDER_SCALE}px`
        canvas.style.height = `${viewport.height / RENDER_SCALE}px`

        renderTask = page.render({ canvasContext: ctx, viewport })
        await renderTask.promise
        if (cancelled) return

        onPageRendered?.({
          widthPts: viewportAtScale1.width,
          heightPts: viewportAtScale1.height,
          // CSS-displayed width divided by point width = what HighlightOverlay
          // and MarginRail multiply raw bbox coordinates by. this always
          // equals exactly 1.0 today, because canvas.style.width above is
          // set to viewportAtScale1.width (no zoom control exists yet) - it's
          // written as a division rather than hardcoded so that when a zoom
          // feature sets canvas.style.width to viewportAtScale1.width * zoom
          // instead, this keeps computing the right value with no changes
          // needed in HighlightOverlay or MarginRail.
          displayScale: viewport.width / RENDER_SCALE / viewportAtScale1.width,
          numPages: doc.numPages,
        })
      } catch (err) {
        if (!cancelled) setError(err.message)
      }
    }

    render()
    return () => {
      cancelled = true
      renderTask?.cancel()
    }
  }, [doc, pageNumber, onPageRendered])

  if (error) {
    return <div className="pdf-canvas-error">Couldn't render this page: {error}</div>
  }

  return <canvas ref={canvasRef} className="pdf-canvas" />
}