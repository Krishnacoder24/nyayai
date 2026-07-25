/*
  the margin rail - a thin strip alongside the page with one tick per error,
  positioned at its proportional vertical spot, colored by error_type. this
  is the one deliberately distinctive element in the viewer: it's how a
  lawyer marks up a physical printout in the margin, not a decorative
  flourish - and it doubles as a way to jump straight to an error without
  scrolling past it.
*/
export default function MarginRail({ errors, pageHeightPts, activeErrorIndex, onSelect }) {
  if (!pageHeightPts) return null

  return (
    <div className="margin-rail">
      {errors.map((error, i) => {
        const [, y0, , y1] = error.bbox
        const midY = (y0 + y1) / 2
        // clamp to [0, 100] - a bbox with a y-coordinate outside the page
        // (same malformed-bbox risk HighlightOverlay guards against)
        // would otherwise push the tick above or below the visible rail.
        const topPercent = Math.min(100, Math.max(0, (midY / pageHeightPts) * 100))
        const isActive = i === activeErrorIndex

        return (
          <button
            key={i}
            type="button"
            className={`margin-tick${isActive ? ' margin-tick--active' : ''}`}
            style={{ top: `${topPercent}%`, '--tick-color': error.highlight_color }}
            title={`${error.error_type}: ${error.text}`}
            onClick={() => onSelect?.(i)}
          />
        )
      })}
    </div>
  )
}