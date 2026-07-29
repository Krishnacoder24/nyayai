"""
manual OCR smoke test against a REAL pdf - typically a real scanned FIR,
to eyeball extract()'s output (page numbers, bboxes, native vs. surya
source) on documents nowhere near as clean as the synthetic fixture in
tests/test_ocr.py.

this is NOT part of the automated suite (pytest never collects it - it's
outside tests/, and it exits via sys.argv handling rather than asserting
anything). it used to live at tests/test_ocr.py; moved here as part of
Issue #50 so that path could become a real pytest file instead, without
losing this manual tool.

usage: make test-ocr FILE=path/to/real_scanned_fir.pdf
"""

import sys
import os

# make sure project root is on the path regardless of where this is called from
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ocr.pipeline import extract

if len(sys.argv) < 2:
    print("usage: make test-ocr FILE=path/to/file.pdf")
    sys.exit(1)

# resolve file path relative to project root, not cwd
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
pdf_path = os.path.join(project_root, sys.argv[1])

if not os.path.exists(pdf_path):
    print(f"error: file not found: {pdf_path}")
    sys.exit(1)

spans = extract(pdf_path)

print(f"total spans: {len(spans)}")
print(f"sources: {set(s.source for s in spans)}")
print()
for s in spans[:10]:
    print(f"  page={s.page_no} source={s.source}")
    print(f"  bbox={s.bbox}")
    print(f"  text={repr(s.text[:60])}")
    print()