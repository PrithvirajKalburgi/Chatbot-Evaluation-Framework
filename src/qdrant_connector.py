# src/qdrant_connector.py
from qdrant_client import QdrantClient
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def fetch_relevant_data(query_embedding: list) -> dict:
    """
    Fetch relevant data from Qdrant collections.
    Args:
        query_text (str): User query to search for relevant context
    
    Returns:
        dict: Dictionary with collection name and retrieved text
    """
    for collection in QDRANT_COLLECTION:
        search_result = client.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=3  # Change this limit based on your requirement (e.g., how many chunks you want)
        )
        if search_result:
            payload = search_result[0].payload  # Get the first relevant result
            return {"collection": collection, "text": payload.get("node_content", {}).get("text", ""), "embedding": search_result[0].vector}
    return {"collection": None, "text": "", "embedding": []}
