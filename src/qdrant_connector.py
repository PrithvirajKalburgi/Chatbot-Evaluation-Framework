# src/qdrant_connector.py
from qdrant_client import QdrantClient
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTIONS

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def fetch_relevant_data(query_text: str) -> dict:
    """
    Fetch relevant data from Qdrant collections.
    Args:
        query_text (str): User query to search for relevant context
    
    Returns:
        dict: Dictionary with collection name and retrieved text
    """
    for collection in QDRANT_COLLECTIONS:
        search_result = client.search(
            collection_name=collection,
            query_vector=client.embeddings.encode(query_text),
            limit=3  # Change this limit based on your requirement (e.g., how many chunks you want)
        )
        if search_result:
            payload = search_result[0].payload  # Get the first relevant result
            return {"collection": collection, "text": payload.get("node_content", {}).get("text", "")}
    return {"collection": None, "text": ""}
