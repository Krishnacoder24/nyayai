"""
GET /health - basic liveness. also reports whether Qdrant is reachable,
since citation checking silently degrades to "skip" when it's down
(see rules/citation_checker.py) - useful to surface that here rather than
only discovering it mid-analysis.
"""

from fastapi import APIRouter
from qdrant_client import QdrantClient

from config.settings import settings

router = APIRouter()


@router.get("/health")
def health():
    qdrant_ok = _check_qdrant()
    return {
        "status": "ok",
        "qdrant": "reachable" if qdrant_ok else "unreachable",
    }


def _check_qdrant() -> bool:
    try:
        client = QdrantClient(url=settings.qdrant_url)
        client.get_collections()
        return True
    except Exception:
        return False