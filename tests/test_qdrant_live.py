import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from corpus.schemas import Passage
from corpus.embeddings import PassageEmbedder 

# Update this if your ingest_corpus.py named the collection something else
COLLECTION_NAME = "legal_corpus" 

@pytest.fixture(scope="module")
def qdrant():
    """
    Connects to the live local Qdrant instance running via Docker.
    Default port for Qdrant is 6333.
    """
    client = QdrantClient(host="localhost", port=6333)
    yield client
    # No teardown needed; we want to leave your data intact.

def test_qdrant_is_running_and_collection_exists(qdrant):
    """Verifies Qdrant is up and the target collection is created."""
    collections_response = qdrant.get_collections()
    collection_names = [c.name for c in collections_response.collections]
    
    assert COLLECTION_NAME in collection_names, \
        f"Could not find '{COLLECTION_NAME}'. Available collections: {collection_names}"

def test_collection_has_points(qdrant):
    """Verifies that the points we ingested are actually stored."""
    collection_info = qdrant.get_collection(collection_name=COLLECTION_NAME)
    
    assert collection_info.points_count > 0, "The collection exists but has 0 points!"
    print(f"\nFound {collection_info.points_count} points in '{COLLECTION_NAME}'.")

def test_point_payload_schema(qdrant):
    """
    Retrieves a single point from the live database without doing a vector search 
    and checks if the metadata matches our Passage schema.
    """
    records, next_page_offset = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=1
    )
    
    assert len(records) == 1, "Failed to retrieve a point from the database."
    
    point = records[0]
    payload = point.payload
    
    assert payload is not None, "Payload is empty!"
    assert "act" in payload, "Missing 'act' in payload"
    assert "unit_type" in payload, "Missing 'unit_type' in payload"
    assert "number" in payload, "Missing 'number' in payload"
    assert "text" in payload, "Missing operative 'text' in payload"
    
    valid_acts = ["crpc", "bnss", "constitution", "cpc", "bns", "ipc"]
    assert payload["act"].lower() in valid_acts, f"Unknown act: {payload['act']}"


def test_search_by_exact_section(qdrant):
    """
    Tests finding a specific section using pure metadata filtering.
    """
    records, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="act", match=MatchValue(value="bns")),
                FieldCondition(key="number", match=MatchValue(value="303")) 
            ]
        ),
        limit=5
    )
    
    assert len(records) > 0, "Could not find the requested section in the database."
    
    print("\n--- Exact Match Results ---")
    for record in records:
        title = record.payload.get('title', 'No Title')
        # FIX: Access part from the nested metadata dictionary
        part = record.payload.get('metadata', {}).get('part', 'unknown')
        print(f"Found: {record.payload['act'].upper()} Sec {record.payload['number']} ({part}) - {title}")


def reconstruct_full_section(qdrant, collection_name: str, act: str, number: str) -> str:
    """
    Takes an act and section number, queries Qdrant for all matching chunks, 
    and stitches them back together into a properly sorted string.
    """
    records, _ = qdrant.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="act", match=MatchValue(value=act)),
                FieldCondition(key="number", match=MatchValue(value=number))
            ]
        ),
        limit=100, 
        with_payload=True
    )
    
    if not records:
        return ""

    # Helper function to extract part from nested metadata
    def get_part(record):
        return record.payload.get("metadata", {}).get("part", "unknown")
    
    # Sorting logic to put the main operative text at the top
    def sort_key(record):
        part = get_part(record)
        if part in ["body", "body_intro"]:
            return (0, part)
        elif part.startswith("clause"):
            return (1, part)
        else:
            return (2, part)

    records.sort(key=sort_key)
    
    reconstructed_text = [f"--- Full Context for {act.upper()} Section {number} ---"]
    
    for record in records:
        part = get_part(record)
        text = record.payload.get("text", "")
        
        if part in ["body", "body_intro"]:
            reconstructed_text.append(f"{text}")
        else:
            clean_label = part.replace("_", " ").title()
            reconstructed_text.append(f"[{clean_label}]: {text}")
            
    return "\n".join(reconstructed_text)


def test_semantic_search_for_theft(qdrant):
    """
    Tests searching by meaning, then reconstructs the full section context 
    based on the top hit.
    """
    embedder = PassageEmbedder()
    
    dummy_query_passage = Passage(
        act="query",
        unit_type="query",
        number="0",
        title="query",
        status="active",
        text="what is the punishment for theft",
        metadata={}
    )
    
    query_vector = embedder.embed_passages([dummy_query_passage])[0]
    
    search_result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3 
    ).points
    
    assert len(search_result) > 0, "Semantic search returned no results."
    
    top_hit = search_result[0]
    hit_act = top_hit.payload.get('act', '')
    hit_number = top_hit.payload.get('number', '')
    hit_score = top_hit.score
    
    # FIX: Access part from the nested metadata dictionary
    hit_part = top_hit.payload.get('metadata', {}).get('part', 'unknown')
    
    print(f"\nTop Hit [Score: {hit_score:.4f}]: {hit_act.upper()} Section {hit_number} (Matched on: {hit_part})\n")
    
    full_context = reconstruct_full_section(
        qdrant=qdrant, 
        collection_name=COLLECTION_NAME, 
        act=hit_act, 
        number=hit_number
    )
    
    print(full_context)