"""
the ONLY way rules code should touch Qdrant. rules/citation_checker.py
calls corpus.search, never the Qdrant client directly - that's what keeps
citation checking, future semantic retrieval, and future repeal mapping
all working off the same collection without rules code needing to know
Qdrant's API.

exact-field filter, no vector similarity, no model loading - citation
checking works even when the ML token-classification model isn't loaded.
"""

import json
from functools import lru_cache
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from corpus.uploader import COLLECTION_NAME, get_client

SCHEDULE_DATA_DIR = Path(__file__).resolve().parent / "data"
SCHEDULE_FILES = {
    "CRPC": SCHEDULE_DATA_DIR / "crpc_schedule.json",
    "BNSS": SCHEDULE_DATA_DIR / "bnss_schedule.json",
}


@lru_cache(maxsize=None)
def _load_schedule(act: str) -> dict:
    path = SCHEDULE_FILES.get(act.strip().upper())
    if path is None or not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def lookup_schedule_entry(number: str, act: str) -> list[dict] | None:
    """exact lookup into the First Schedule (CrPC) / equivalent (BNSS)
    offence-classification table: cognizable/bailable status and which
    court tries it, for a given section number."""
    schedule = _load_schedule(act)
    return schedule.get(number)


def lookup_section(number: str, act: str, client: QdrantClient | None = None) -> dict | None:
    """exact citation lookup: does this section/article exist, and is it active?"""
    client = client or get_client()
    normalized_act = act.strip().lower()

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query_filter=Filter(
            must=[
                FieldCondition(key="number", match=MatchValue(value=number)),
                FieldCondition(key="act", match=MatchValue(value=normalized_act)),
            ]
        ),
        limit=1,
        with_payload=True,
    )

    if not result.points:
        return None

    return result.points[0].payload