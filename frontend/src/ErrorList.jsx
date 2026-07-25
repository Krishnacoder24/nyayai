const TYPE_LABELS = {
  spelling: 'Spelling',
  grammar: 'Grammar',
  citation: 'Citation',
  entity: 'Entity consistency',
}

export default function ErrorList({ report, activeErrorIndex, onSelect }) {
  return (
    <aside className="error-list">
      <div className="error-list-summary">
        <strong>{report.total_errors}</strong> issue{report.total_errors === 1 ? '' : 's'} found
        <div className="error-list-counts">
          {Object.entries(report.errors_by_type).map(([type, count]) => (
            <span key={type} className="error-count-chip" style={{ '--chip-color': errorColor(report, type) }}>
              {count} {TYPE_LABELS[type] || type}
            </span>
          ))}
        </div>
      </div>

      <ul className="error-list-items">
        {report.errors.map((error, i) => (
          <li key={i}>
            <button
              type="button"
              className={`error-list-item${i === activeErrorIndex ? ' error-list-item--active' : ''}`}
              onClick={() => onSelect?.(i)}
            >
              <span className="error-swatch" style={{ background: error.highlight_color }} />
              <span className="error-list-item-body">
                <span className="error-list-item-text">{error.text}</span>
                <span className="error-list-item-meta">
                  p.{error.page_no} · {TYPE_LABELS[error.error_type] || error.error_type}
                </span>
                {error.suggestion && <span className="error-list-item-suggestion">{error.suggestion}</span>}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </aside>
  )
}

function errorColor(report, type) {
  const match = report.errors.find((e) => e.error_type === type)
  return match?.highlight_color || '#999'
}