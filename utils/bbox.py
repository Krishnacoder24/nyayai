"""
bbox helpers used across the project.

all bboxes are (x0, y0, x1, y1) tuples in pdfplumber coordinate space:
  - origin top-left
  - x increases rightward
  - y increases downward
  - units are PDF points (1 pt = 1/72 inch)
"""


def merge(bboxes: list[tuple]) -> tuple:
    """
    merges a list of bboxes into one tight bbox that covers all of them.
    used in postprocess.py to build one span bbox from multiple token bboxes.
    """
    if not bboxes:
        raise ValueError("cannot merge empty list of bboxes")
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return (x0, y0, x1, y1)


def overlaps(a: tuple, b: tuple) -> bool:
    """
    returns True if two bboxes overlap.
    touching edges (shared border) are NOT considered overlapping.
    used in pipeline/deduplicate.py.
    """
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def area(bbox: tuple) -> float:
    """
    returns the area of a bbox in square PDF points.
    """
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def intersection(a: tuple, b: tuple) -> tuple | None:
    """
    returns the intersection bbox of two bboxes, or None if they don't overlap.
    """
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def iou(a: tuple, b: tuple) -> float:
    """
    intersection over union — standard metric for how much two bboxes overlap.
    returns 0.0 if no overlap, 1.0 if identical.
    used for smarter deduplication than simple overlap check.
    """
    inter = intersection(a, b)
    if inter is None:
        return 0.0
    inter_area = area(inter)
    union_area = area(a) + area(b) - inter_area
    if union_area == 0.0:
        return 0.0
    return inter_area / union_area


def contains(outer: tuple, inner: tuple) -> bool:
    """
    returns True if outer bbox fully contains inner bbox.
    """
    return (outer[0] <= inner[0] and outer[1] <= inner[1] and
            outer[2] >= inner[2] and outer[3] >= inner[3])


def scale(bbox: tuple, sx: float, sy: float) -> tuple:
    """
    scales a bbox by sx horizontally and sy vertically.
    used when converting between coordinate spaces (e.g. PDF points to pixels).
    """
    return (bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy)