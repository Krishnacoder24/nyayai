"""
Diagnostic probe: finds the First Schedule pages in bnss.pdf and
prints character x-coordinates and raw text so we can identify the
exact column cut-points before writing the real parser.
"""
import pdfplumber
from pathlib import Path

PDF_PATH = Path(__file__).resolve().parent.parent / "corpus" / "sources" / "bnss" / "bnss.pdf"

def main():
    with pdfplumber.open(PDF_PATH) as pdf:
        print(f"Total pages: {len(pdf.pages)}")

        schedule_start = None
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if "THE FIRST SCHEDULE" in text and schedule_start is None:
                schedule_start = i
                print(f"\n>>> FIRST SCHEDULE starts at page index {i} (page {i+1})\n")

        if schedule_start is None:
            print("ERROR: 'THE FIRST SCHEDULE' not found anywhere in the PDF!")
            return

        # Inspect up to 3 pages of the schedule
        for page_idx in range(schedule_start, min(schedule_start + 3, len(pdf.pages))):
            page = pdf.pages[page_idx]
            print(f"\n{'='*70}")
            print(f"PAGE {page_idx + 1}  (size: {page.width:.1f} x {page.height:.1f})")
            print(f"{'='*70}")

            # Check for drawn lines/rects (table boundaries)
            print(f"  Lines on page: {len(page.lines)}")
            print(f"  Rects on page: {len(page.rects)}")

            # Group chars by row (rounded top coordinate)
            chars = [c for c in page.chars if c.get("text", "").strip()]
            rows: dict[int, list] = {}
            for ch in chars:
                key = round(ch["top"])
                rows.setdefault(key, []).append(ch)

            print(f"\n  First 30 rows (top → text | x0 positions):")
            for top in sorted(rows.keys())[:30]:
                row_chars = sorted(rows[top], key=lambda c: c["x0"])
                text = "".join(c["text"] for c in row_chars)
                x_positions = [round(c["x0"]) for c in row_chars[::3]]  # sample every 3rd char
                print(f"  y={top:4d}  x_start={round(row_chars[0]['x0']):4d}  '{text[:80]}'")

            # Print unique x0 start positions (column anchors) from first 40 rows
            print(f"\n  Unique x0 start positions per row (column anchors):")
            col_starts = set()
            for top in sorted(rows.keys())[:40]:
                row_chars = sorted(rows[top], key=lambda c: c["x0"])
                # Find x0 of first char of each "word cluster" (gap > 10px)
                prev_x1 = 0
                for ch in row_chars:
                    if ch["x0"] - prev_x1 > 10:
                        col_starts.add(round(ch["x0"] / 5) * 5)  # round to nearest 5
                    prev_x1 = ch["x1"]
            print(f"  {sorted(col_starts)}")

if __name__ == "__main__":
    main()
