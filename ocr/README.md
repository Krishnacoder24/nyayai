# phase 1 — ocr pipeline (`feature/ocr`)

## what this phase does

takes any pdf (native text or scanned) and returns a flat list of `LineSpan`
objects — one per line of text — each carrying the line's text content, which
page it came from, and its exact bounding box coordinates on that page.

single public API:
```python
from ocr.pipeline import extract

spans = extract("some_legal_doc.pdf")
# returns list[LineSpan], sorted by page then top-to-bottom
```

---

## files built

```
ocr/
├── __init__.py
├── tokens.py            LineSpan dataclass - the base unit for everything downstream
├── native_extractor.py  NativeExtractor - pdfplumber, for pdfs with a text layer
├── surya_extractor.py   SuryaExtractor - surya 0.9.3, for scanned/image pdfs
├── router.py            route() - one pass to extract native spans + find scanned pages
└── pipeline.py          extract() - the only function the rest of the project calls
```

---

## what i achieved

- working OCR pipeline that handles native pdfs, scanned pdfs, and mixed pdfs
  (some pages native, some scanned) in one function call
- tested against a real scanned marksheet - surya correctly read institution name,
  student ID, course codes, grades, SPI/CPI
- bboxes are real measured coordinates in both extractors, no approximation anywhere
- surya loads model weights once and reuses across all pages, not per-page
- pdf opens once for rasterization, pages fed to surya in chunks of 4 to stay
  within 6gb vram on mixed/large documents
- router does native extraction and page classification in one pass - no redundant
  file opens

---

## what i did wrong (and fixed)

**WordToken was the wrong base unit.**
started with `WordToken` - one object per word. made sense on paper since
highlighting errors per-word feels natural. problem: pdfplumber gives real
word-level bboxes but surya 0.9.3 only gives line-level bboxes. to get
word-level out of surya we were splitting each line proportionally by character
count - pure approximation. switched to `LineSpan` (one object per line) which
maps directly to what both extractors actually produce. no approximation anywhere.
lesson: design your data model around what the tools actually give you, not what
you wish they gave you.

**surya was opening and closing the pdf on every page.**
`_render_page()` was called once per page, each time doing
`pdfium.PdfDocument(pdf_path)` → render → `doc.close()`. 20 page FIR = 20 pdf
parses. fixed by rendering all pages in one `_render_pages()` call with the doc
open for the full loop.

**then went too far and batched everything at once.**
after fixing the per-page open, batched all images into one surya call. looks
efficient but detection holds activation memory for every image simultaneously -
on a 6gb card a 30 page scanned judgment would OOM. fixed by chunking in groups
of 4 pages per surya call. get the pdf-open win without the vram bomb.

**router was doing redundant work.**
`router.py` called `has_text_layer()` which opened the pdf and read every page
to count characters. then `pipeline.py` called `NativeExtractor.extract()` which
opened the pdf again and read every page to get spans. same work twice. fixed by
making router return the native spans directly - extract once, classify by char
count, pass spans straight to pipeline.

---

## what i learned

**surya 0.9.3 vs 0.9.3 + transformers 4.57.6 = broken.**
`KeyError: 'encoder'` on `RecognitionPredictor.__init__` is a transformers
version incompatibility. `SuryaOCRConfig.__init__` expects `encoder` in kwargs
when `transformers` calls it during `to_diff_dict()`, which changed in newer
transformers versions. fix: downgrade to `transformers==4.48.0`. surya 0.9.3 was
built against a specific transformers era and doesn't survive upgrades past ~4.50.

**pdfplumber coordinate system.**
`top` and `bottom` in pdfplumber are measured from the top-left of the page (y
increases downward, like screen coordinates). raw PDF spec and some other libraries
measure from bottom-left (y increases upward). i mapped `top → y0`, `bottom → y1`
directly. if highlights ever appear vertically flipped, this is the first place
to check.

**surya batch sizes matter a lot on laptop gpus.**
default batch sizes in 0.9.3 are tuned for big server cards. on the 4050 you need
`RECOGNITION_BATCH_SIZE=32 DETECTOR_BATCH_SIZE=4` as env vars before running or
you'll OOM on the model load itself, not even on inference.

**ocr.tokens is a module path, not a package.**
`from ocr.tokens import LineSpan` only works if you run python from the project
root (`~/nyayai/`), not from inside the `ocr/` folder. python resolves the import
relative to wherever you launched from. `__init__.py` makes `ocr/` an explicit
package to avoid edge case import failures as the project grows.

**`has_text_layer()` is now dead code.**
router no longer calls it since we extract first and classify by char count on the
results. left it in `native_extractor.py` for now in case it's useful for
debugging, but it's not called anywhere in the pipeline.

---

## env vars to set before running surya

```bash
RECOGNITION_BATCH_SIZE=32 DETECTOR_BATCH_SIZE=4
```

add these to your `.env` or set them in the shell before any command that touches
scanned pdfs. without them, surya will likely OOM on first model load.

---

## what comes next (phase 2)

`feature/model` — takes the `list[LineSpan]` this phase produces and runs
InLegalBERT on it to detect spelling errors, grammar errors, and wrong IPC/BNS
citations. first file: `model/preprocess.py`.