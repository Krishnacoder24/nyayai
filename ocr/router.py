from pathlib import Path
import logging

import pymupdf as fitz
from pdf2image import convert_from_path

from extractor import NativeExtractor, SuryaExtractor, WordToken

log = logging.getLogger(__name__)

CHAR_THRESHOLD = 500
DPI = 200


def process_pdf(pdf_path, languages=None):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"file not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    if doc.page_count == 0:
        raise ValueError("pdf has no pages")

    native = NativeExtractor()
    surya = SuryaExtractor(languages or ["en", "hi"])

    log.info(f"processing {pdf_path.name} - {doc.page_count} pages")

    # check which pages need ocr
    char_counts = [native.char_count(doc.load_page(i)) for i in range(doc.page_count)]
    needs_ocr = [i for i, c in enumerate(char_counts) if c < CHAR_THRESHOLD]

    log.info(f"{len(needs_ocr)} pages need ocr, {doc.page_count - len(needs_ocr)} native")

    # rasterise only pages that need ocr
    images = {}
    if needs_ocr:
        all_imgs = convert_from_path(str(pdf_path), dpi=DPI)
        images = {i: all_imgs[i] for i in needs_ocr if i < len(all_imgs)}

    pages = []
    for i in range(doc.page_count):
        if i not in needs_ocr:
            tokens = native.extract_page(doc.load_page(i), i)
        else:
            try:
                tokens = surya.run_ocr(images[i], i)
            except Exception as e:
                log.warning(f"surya failed on page {i}: {e}, falling back to native")
                tokens = native.extract_page(doc.load_page(i), i)
        pages.append(tokens)

    doc.close()
    log.info(f"done - {sum(len(p) for p in pages)} words extracted")
    return pages


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("usage: python ocr/router.py path/to/file.pdf")
        sys.exit(1)

    pages = process_pdf(sys.argv[1])
    for i, page in enumerate(pages):
        print(f"\npage {i+1} — {len(page)} words")
        for token in page[:5]:
            print(f"  {token.source} | '{token.text}' | {[round(x,1) for x in token.bbox]}")