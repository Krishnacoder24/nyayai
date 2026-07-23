/*
  draws one highlight box per error on the current page, positioned by
  scaling each error's PDF-point bbox by displayScale (from PdfCanvas).
  no y-flip - see PdfCanvas.jsx's comment for why.
*/
export default function HighlightOverlay({ errors, displayScale, activeErrorIndex, onSelect }) {
  if (!displayScale) return null

  return (
    <div className="highlight-overlay">
      {errors.map((error, i) => {
        const [x0, y0, x1, y1] = error.bbox
        const isActive = i === activeErrorIndex

        // a malformed ErrorSpan (bad OCR bbox, a model/rules bug, etc.)
        // can hand us x1 < x0 or y1 < y0. left/top use Math.min so the box
        // still anchors at the correct corner regardless of point order,
        // and width/height are clamped to >= 0 so a reversed bbox renders
        // as a zero-size (rather than negative, which some browsers just
        // silently drop the element for) box instead of disappearing with
        // no signal that something upstream produced bad coordinates.
        const left = Math.min(x0, x1) * displayScale
        const top = Math.min(y0, y1) * displayScale
        const width = Math.max(0, Math.abs(x1 - x0) * displayScale)
        const height = Math.max(0, Math.abs(y1 - y0) * displayScale)

        return (
          <button
            key={i}
            type="button"
            className={`highlight-box${isActive ? ' highlight-box--active' : ''}`}
            style={{
              left,
              top,
              width,
              height,
              '--highlight-color': error.highlight_color,
            }}
            title={error.suggestion || error.text}
            onClick={() => onSelect?.(i)}
          />
        )
      })}
    </div>
  )
}