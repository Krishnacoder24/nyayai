/*
  mock data shaped exactly like renderer/report.py's build_report() output -
  same keys (source_filename, total_pages, total_errors, errors_by_type,
  errors[]), same per-error fields (bbox, highlight_color, confidence, etc).

  the point: api.js's real implementation (once api/routes/jobs.py exists)
  just needs to return this same shape, and nothing else in the app changes.
*/

export function buildMockReport(sourceFilename) {
  const errors = [
    {
      text: 'Section 302 IPC',
      error_type: 'citation',
      page_no: 0,
      x0: 120, y0: 180, x1: 260, y1: 198,
      bbox: [120, 180, 260, 198],
      suggestion: 'verify Section 302 IPC exists and is active - consider Section 103 BNS',
      confidence: 0.95,
      highlight_color: '#FF4444',
      explanation: 'no section numbered 999 was found under IPC in the corpus',
    },
    {
      text: 'recieved',
      error_type: 'spelling',
      page_no: 0,
      x0: 90, y0: 260, x1: 160, y1: 278,
      bbox: [90, 260, 160, 278],
      suggestion: 'received',
      confidence: 0.88,
      highlight_color: '#FFD700',
      // spelling explanations aren't populated yet (Issue #17) - left
      // blank on purpose to exercise ErrorPopover's placeholder text.
      explanation: '',
    },
    {
      text: 'has been went',
      error_type: 'grammar',
      page_no: 0,
      x0: 200, y0: 340, x1: 320, y1: 358,
      bbox: [200, 340, 320, 358],
      suggestion: 'has gone',
      confidence: 0.81,
      highlight_color: '#FF8C00',
      explanation: '',
    },
    {
      text: 'Rakesh Kumar',
      error_type: 'entity',
      page_no: 0,
      x0: 130, y0: 480, x1: 260, y1: 498,
      bbox: [130, 480, 260, 498],
      suggestion: 'should be "Ramesh Kumar"',
      confidence: 0.92,
      highlight_color: '#8A2BE2',
      explanation: '92% match to "Ramesh Kumar", the more frequent mention in this document',
    },
    {
      text: 'Patana',
      error_type: 'entity',
      page_no: 1,
      x0: 150, y0: 220, x1: 220, y1: 238,
      bbox: [150, 220, 220, 238],
      suggestion: 'should be "Patna"',
      confidence: 0.9,
      highlight_color: '#8A2BE2',
      explanation: '90% match to "Patna", the more frequent mention in this document',
    },
  ]

  const errors_by_type = {}
  for (const e of errors) {
    errors_by_type[e.error_type] = (errors_by_type[e.error_type] || 0) + 1
  }

  return {
    source_filename: sourceFilename,
    total_pages: 2,
    total_errors: errors.length,
    errors_by_type,
    errors,
  }
}