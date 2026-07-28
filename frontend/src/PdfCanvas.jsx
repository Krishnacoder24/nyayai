import { useEffect, useRef, useState } from 'react'
import * as pdfjsLib from 'pdfjs-dist'
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.mjs?url'

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl

// raster resolution is capped as a multiple of the CSS display scale, not
// the PDF's own point size - without this, zooming in with the buttons
// below would render at 1x-ish internal resolution and just look blurry
// once CSS stretched it up. capped at 4 so a large zoom on a big page
// doesn't blow past a sane canvas memory budget.
const MAX_RASTER_SCALE = 4

/*
  renders `pageNumber` of `file` onto a canvas, sized to fill `containerWidth`
  (times `zoom`) rather than a fixed 1-CSS-pixel-per-PDF-point size - a
  Letter/A4 page at 1:1 renders as a small rectangle lost in a mostly-empty
  viewer on any normal-sized monitor. containerWidth comes from App.jsx's
  ResizeObserver on the canvas area; zoom is the user-controlled multiplier
  on top of that fit-to-width baseline (1.0 = fit exactly).

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
export default function PdfCanvas({ file, pageNumber, containerWidth, zoom, onPageRendered }) {
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

        // fitScale fills containerWidth exactly at zoom=1; falls back to
        // 1:1 point-to-CSS-pixel sizing if containerWidth isn't known yet
        // (first paint, before the ResizeObserver in App.jsx has measured
        // anything) rather than flashing a 0-width canvas.
        const fitScale = containerWidth
          ? containerWidth / viewportAtScale1.width
          : 1
        const displayScale = fitScale * (zoom || 1)

        const rasterScale = Math.min(displayScale, MAX_RASTER_SCALE)
        const viewport = page.getViewport({ scale: rasterScale })

        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        canvas.width = viewport.width
        canvas.height = viewport.height
        canvas.style.width = `${viewportAtScale1.width * displayScale}px`
        canvas.style.height = `${viewportAtScale1.height * displayScale}px`

        renderTask = page.render({ canvasContext: ctx, viewport })
        await renderTask.promise
        if (cancelled) return

        onPageRendered?.({
          widthPts: viewportAtScale1.width,
          heightPts: viewportAtScale1.height,
          // CSS-displayed width divided by point width - what HighlightOverlay
          // and MarginRail multiply raw bbox coordinates by to land on the
          // right on-screen pixel regardless of fit-to-width sizing or zoom.
          displayScale,
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
  }, [doc, pageNumber, containerWidth, zoom, onPageRendered])

  if (error) {
    return <div className="pdf-canvas-error">Couldn't render this page: {error}</div>
  }

  return <canvas ref={canvasRef} className="pdf-canvas" />
}