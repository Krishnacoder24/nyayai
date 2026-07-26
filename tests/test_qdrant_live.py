import pytest
from qdrant_client import QdrantClient

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
    
    # Based on your terminal output, you uploaded thousands of points.
    # This just checks that the database isn't empty.
    assert collection_info.points_count > 0, "The collection exists but has 0 points!"
    print(f"\nFound {collection_info.points_count} points in '{COLLECTION_NAME}'.")

def test_point_payload_schema(qdrant):
    """
    Retrieves a single point from the live database without doing a vector search 
    and checks if the metadata matches our Passage schema.
    """
    # .scroll() fetches records directly by ID order without needing a query vector
    records, next_page_offset = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=1
    )
    
    assert len(records) == 1, "Failed to retrieve a point from the database."
    
    point = records[0]
    payload = point.payload
    
    # Assert the required fields from your Section/Passage schema exist
    assert payload is not None, "Payload is empty!"
    assert "act" in payload, "Missing 'act' in payload"
    assert "unit_type" in payload, "Missing 'unit_type' in payload"
    assert "number" in payload, "Missing 'number' in payload"
    assert "text" in payload, "Missing operative 'text' in payload"
    
    # Ensure the acts match the ones in your ingest script
    valid_acts = ["crpc", "bnss", "constitution", "cpc", "bns", "ipc"]
    assert payload["act"].lower() in valid_acts, f"Unknown act: {payload['act']}"