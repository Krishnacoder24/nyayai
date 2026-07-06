"""
extracts text from pdfs that already have a text layer (most digitally
created pdfs - word exports, e-filed court documents, etc).

this does NOT do OCR. if you run this on a scanned image-only pdf, you'll
get back an empty list or near empty, since there's no text layer to read.
that's expected - router.py (wednesday) will decide when to fall back to
surya for those pages.
"""

import pdfplumber

from ocr.tokens import WordToken


class NativeExtractor:
    def extract(self, pdf_path: str) -> list[WordToken]:
        tokens = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_no, page in enumerate(pdf.pages):
                words = page.extract_words()

                for w in words:
                    token = WordToken(
                        text=w["text"],
                        page_no=page_no,
                        source="native",
                        x0=w["x0"],
                        y0=w["top"],
                        x1=w["x1"],
                        y1=w["bottom"],
                    )
                    if token.is_valid():
                        tokens.append(token)

        return tokens

    def has_text_layer(self, pdf_path: str, min_chars_per_page: int = 20) -> dict[int, bool]:
        """
        per page check: does this page have enough native text to skip OCR?
        router.py will use this to decide native vs surya per page.
        returns a dict of {page_no: bool}
        """
        result = {}

        with pdfplumber.open(pdf_path) as pdf:
            for page_no, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                result[page_no] = len(text.strip()) >= min_chars_per_page

        return result