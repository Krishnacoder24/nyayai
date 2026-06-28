
from dataclasses import dataclass
import logging

log = logging.getLogger


@dataclass
class WordToken:
    text: str
    page_no: int
    source: str
    x0: float
    y0: float
    x1: float    
    y1: float

    @property
    def bbox(self):
        return [self.x0, self.y0, self.x1, self.y1] #returns as a list
    
    def is_valid(self):
        return self.text.strip() != '' and self.x1 - self.x0 > 0 and self.y1 - self.y0 > 0
    


class NativeExtractor:
    def extract_page(self,page,page_no):
        tokens = []
        for x0,y0,x1,y1,text in page.get_text("words"):
            t = WordToken(text,page_no,'native',x0,y0,x1,y1)
            if t.is_valid():
                tokens.append(t)
        return tokens
    
    def char_count(self,page):
        return len(page.extract_text().strip())
    

class SuryaExtractor:
    
    RECOGNITION_BATCH = 32
    DETECTION_BATCH = 4

    def __init__(self,languages=None):
        self.language = languages or ['eng', 'hi']
        self._rec = None
        self._det = None

    def load_models(self):
        if self._rec is None:
            return
        log.info("Loading Surya OCR models...")
        from surya.recoginition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        self._rec = RecognitionPredictor()
        self._det = DetectionPredictor()
        log.info("Surya OCR models loaded successfully.")

    def run_ocr(self, pil_image, page_no):
        self.load_models()
        results = self._rec(
            images=[pil_image],
            langs=[self.languages],
            det_predictor=self._det,
            recognition_batch_size=self.RECOGNITION_BATCH,
            detection_batch_size=self.DETECTION_BATCH,
        )
        tokens = []
        for line in results[0].text_lines:
            if not line.text.strip():
                continue
            b = line.bbox
            tokens += self._split_line(line.text, b[0], b[1], b[2], b[3], page_no)
        return tokens
    
    def _split_line(self, text, lx0, ly0, lx1, ly1, page_no):
        # surya gives line bbox, splitting into words proportionally
        words = text.split()
        if not words:
            return []
        total = sum(len(w) for w in words)
        width = lx1 - lx0
        x = lx0
        tokens = []
        for w in words:
            ww = width * (len(w) / total)
            t = WordToken(w, page_no, "surya", x, ly0, x + ww, ly1)
            if t.is_valid():
                tokens.append(t)
            x += ww
        return tokens