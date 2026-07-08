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