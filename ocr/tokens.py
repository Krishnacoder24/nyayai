from dataclasses import dataclass


@dataclass
class WordToken:
    text: str
    page_no: int
    source: str  # "native" or "surya"
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def bbox(self):
        return (self.x0, self.y0, self.x1, self.y1)

    def is_valid(self):
        return bool(self.text.strip()) and self.x1 > self.x0 and self.y1 > self.y0