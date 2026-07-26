"""
pushes embedded passages to the single "legal_corpus" Qdrant collection.
every act lives in the same collection - filtering by "act" + "number" is
what makes exact citation lookup fast, and one collection (rather than
one per act) is what makes future cross-act features possible (repeal
mapping, hybrid search) without a schema migration later.
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    PayloadSchemaType,
)

from corpus.schemas import Passage
from config.settings import settings

COLLECTION_NAME = "legal_corpus"
VECTOR_SIZE = 768  # InLegalBERT hidden size

# Namespace for deterministic UUID v5 generation
NAMESPACE_LEGAL = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def get_client(url: str | None = None) -> QdrantClient:
    return QdrantClient(url=url or settings.qdrant_url)


def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    # keyword indexes on the two fields citation lookup filters by
    client.create_payload_index(COLLECTION_NAME, "number", PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION_NAME, "act", PayloadSchemaType.KEYWORD)


def drop_act(client: QdrantClient, act: str) -> None:
    """used by --force: removes existing points for one act only, rest of the collection is untouched."""
    normalized_act = act.strip().lower()

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="act", match=MatchValue(value=normalized_act))]
        ),
    )


def upload_passages(client: QdrantClient, passages: list[Passage], vectors: list[list[float]]) -> int:
    points = []
    for passage, vector in zip(passages, vectors):
        normalized_act = passage.act.strip().lower()

        payload = {
            "act": normalized_act,
            "unit_type": passage.unit_type,
            "number": passage.number,
            "title": passage.title,
            "status": passage.status,
            "text": passage.text,
            "metadata": passage.metadata,  # chapter/part/effective_date/replaced_by etc.
        }

        # Deterministic ID based on act, unit_type, number, and text content.
        # This guarantees upserts overwrite existing passages instead of duplicating them.
        unique_key = f"{normalized_act}:{passage.unit_type}:{passage.number}:{passage.text}"
        point_id = str(uuid.uuid5(NAMESPACE_LEGAL, unique_key))

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)