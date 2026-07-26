import re
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from corpus.schemas import Passage
from corpus.embeddings import PassageEmbedder 

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

    def get_part(record):
        return record.payload.get("metadata", {}).get("part", "unknown")
    
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


# Regex pattern to catch explicit section queries like "ipc 76", "section 303 of bns", etc.
SECTION_QUERY_REGEX = re.compile(
    r'\b(?:section|sec\.?)\s*([0-9]+[A-Za-z]*)\b.*?\b(ipc|bns|bnss|cpc|crpc|constitution)\b|'
    r'\b(ipc|bns|bnss|cpc|crpc|constitution)\b.*?\b(?:section|sec\.?)\s*([0-9]+[A-Za-z]*)\b|'
    r'\b(ipc|bns|bnss|cpc|crpc|constitution)\b\s*([0-9]+[A-Za-z]*)\b',
    re.IGNORECASE
)

def smart_legal_search(qdrant, collection_name: str, embedder, query_text: str):
    """
    Hybrid search router:
    1. Extracts explicit Act and Section references via regex.
    2. Bypasses vector search for direct metadata fetch if matched.
    3. Falls back to vector semantic search otherwise.
    """
    match = SECTION_QUERY_REGEX.search(query_text)
    act, section_num = None, None
    
    if match:
        groups = match.groups()
        if groups[0] and groups[1]:
            section_num, act = groups[0], groups[1].lower()
        elif groups[2] and groups[3]:
            act, section_num = groups[2].lower(), groups[3]
        elif groups[4] and groups[5]:
            act, section_num = groups[4].lower(), groups[5]

    if act and section_num:
        print(f"\n[Hybrid Router]: Explicit reference detected -> Act: {act.upper()}, Section: {section_num}")
        records, _ = qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="act", match=MatchValue(value=act)),
                    FieldCondition(key="number", match=MatchValue(value=section_num))
                ]
            ),
            limit=50,
            with_payload=True
        )
        if records:
            return "exact_match", records
            
    print(f"\n[Hybrid Router]: No explicit section detected. Routing to Semantic Vector Search...")
    dummy_query_passage = Passage(
        act="query", unit_type="query", number="0", title="query", status="active",
        text=query_text, metadata={}
    )
    query_vector = embedder.embed_passages([dummy_query_passage])[0]
    
    search_result = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3
    ).points
    
    return "semantic_match", search_result


def execute_user_query(qdrant, embedder, user_query: str):
    """
    Unified query handler:
    - If a direct section is cited (detected via router regex), it performs the direct metadata lookup (explicit).
    - It then always follows up with or performs semantic vector search behavior based on whether a direct match occurred.
    """
    print(f"\n--- Processing User Query: '{user_query}' ---")
    
    # Run the hybrid router logic
    match_type, results = smart_legal_search(
        qdrant=qdrant, 
        collection_name=COLLECTION_NAME, 
        embedder=embedder, 
        query_text=user_query
    )
    
    if match_type == "exact_match":
        print("\n[Step 1: Direct Section Citation Found via Router]")
        hit_act = results[0].payload.get('act', '')
        hit_number = results[0].payload.get('number', '')
        print(f"Direct Lookup Result -> Act: {hit_act.upper()} Section {hit_number}")
        
        full_context = reconstruct_full_section(qdrant, COLLECTION_NAME, hit_act, hit_number)
        print(full_context)
        
        print("\n[Step 2: Performing Semantic Search Integration for Extended Context]")
        dummy_query_passage = Passage(
            act="query", unit_type="query", number="0", title="query", status="active",
            text=user_query, metadata={}
        )
        query_vector = embedder.embed_passages([dummy_query_passage])[0]
        semantic_results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3
        ).points
        
        if semantic_results:
            top_sem = semantic_results[0]
            sem_act = top_sem.payload.get('act', '')
            sem_number = top_sem.payload.get('number', '')
            sem_score = top_sem.score
            print(f"Semantic Search Companion Hit [Score: {sem_score:.4f}] -> {sem_act.upper()} Section {sem_number}")
            sem_full_context = reconstruct_full_section(qdrant, COLLECTION_NAME, sem_act, sem_number)
            print(sem_full_context)
            
    else:
        print("\n[Step 1 & 2: No Direct Section Cited -> Pure Semantic Search]")
        assert len(results) > 0, "Semantic search returned no results."
        top_hit = results[0]
        hit_act = top_hit.payload.get('act', '')
        hit_number = top_hit.payload.get('number', '')
        hit_score = top_hit.score
        hit_part = top_hit.payload.get('metadata', {}).get('part', 'unknown')
        
        print(f"Semantic Search Top Hit [Score: {hit_score:.4f}] -> {hit_act.upper()} Section {hit_number} (Matched on: {hit_part})")
        
        full_context = reconstruct_full_section(qdrant, COLLECTION_NAME, hit_act, hit_number)
        print(full_context)


def test_user_query_with_explicit_section(qdrant):
    """
    Tests when a user asks a question with an explicit section citation (triggers explicit -> semantic flow).
    """
    embedder = PassageEmbedder()
    user_query = "what does ipc 76 says"
    execute_user_query(qdrant, embedder, user_query)


def test_user_query_with_pure_semantic(qdrant):
    """
    Tests when a user asks a question without a direct section citation (triggers semantic flow only).
    """
    embedder = PassageEmbedder()
    user_query = "what is the punishment for theft"
    execute_user_query(qdrant, embedder, user_query)