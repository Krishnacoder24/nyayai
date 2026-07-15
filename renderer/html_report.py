"""
renders the structured report (report.py) into a single self-contained
HTML file - viewable without the original PDF, the API, or the frontend.
useful for sharing a quick summary without the full app.
"""

from pathlib import Path

from renderer.colors import get_hex


def render_html(report: dict, output_path: Path) -> None:
    rows = "\n".join(_error_row(e) for e in report["errors"]) or _empty_row()
    title_suffix = f' — {report["source_filename"]}' if report.get("source_filename") else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NyayAI error report{title_suffix}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 2rem; color: #1a1a1a; }}
  h1 {{ font-size: 1.4rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
  th {{ background: #f5f5f5; }}
  .swatch {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }}
  .summary {{ color: #555; margin-bottom: 1rem; }}
</style>
</head>
<body>
  <h1>NyayAI Error Report{title_suffix}</h1>
  <p class="summary">{_summary_line(report)}</p>
  <table>
    <thead>
      <tr><th>Type</th><th>Page</th><th>Text</th><th>Suggestion</th><th>Confidence</th></tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def _summary_line(report: dict) -> str:
    if report["total_errors"] == 0:
        return "No errors found."
    parts = [f"{count} {etype}" for etype, count in report["errors_by_type"].items()]
    return f'{report["total_errors"]} total error(s): ' + ", ".join(parts)


def _error_row(error: dict) -> str:
    color = error.get("highlight_color") or get_hex(error["error_type"])
    return (
        "<tr>"
        f'<td><span class="swatch" style="background:{color}"></span>{_escape(error["error_type"])}</td>'
        f'<td>{error["page_no"]}</td>'
        f'<td>{_escape(error["text"])}</td>'
        f'<td>{_escape(error.get("suggestion") or "—")}</td>'
        f'<td>{error["confidence"]:.2f if error["confidence"] is not None else "—"}</td>' # what if confidence is None? should be a float, but just in case.
        "</tr>"
    )


def _empty_row() -> str:
    return '<tr><td colspan="5">No errors found.</td></tr>'


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")