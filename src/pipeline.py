from datetime import datetime, timezone 
from typing import Optional
from pymongo import DESCENDING
from mongodb_connector import fetch_user_query, fetch_retrieved_chunks, store_evaluation, prompt_collection
from embedding_utils import embed_text
from evaluation.accuracy import compute_accuracy
from evaluation.relevance import compute_relevance
from evaluation.hallucination import detect_hallucination
import fire

def evaluate_query_response(query_id: Optional[str] = None, verify_latest: bool = True):
    """
    Evaluate a single query response
    Args:
        query_id: Specific ID to evaluate (None for most recent)
        verify_latest: Double-check we're evaluating the newest document
    """
    query_data = fetch_user_query(query_id)
    
    if not query_data:
        print("No query data found.")
        return
    
    # Extra verification for latest document
    if verify_latest and query_id is None:
        actual_latest = prompt_collection.find_one(
            sort=[('_id', DESCENDING)],
            projection=['_id']
        )
        if actual_latest and query_data['_id'] != actual_latest['_id']:
            print(f"Warning: Found newer document {actual_latest['_id']}, re-evaluating")
            return evaluate_query_response(actual_latest['_id'], verify_latest=False)

    user_query = query_data['user_prompt']['text']
    ai_response = query_data['ai_response']['text']
    retrieved_chunks = fetch_retrieved_chunks(query_data["_id"])
    
    print(f"Evaluating query from {query_data.get('timestamp')} (ID: {query_data['_id']})")

    # Generate embeddings
    query_embed = embed_text(user_query)
    response_embed = embed_text(ai_response)
    chunks_embed = [embed_text(chunk) for chunk in retrieved_chunks]

    metrics = {
        "accuracy": compute_accuracy(
            predicted=ai_response,
            reference_chunks=retrieved_chunks,
            predicted_embedding=response_embed,
            reference_embeddings=chunks_embed
        ),
        "relevance": compute_relevance(
            query_embedding=query_embed,
            response_embedding=response_embed
        ),
        "hallucination": detect_hallucination(
            predicted_response=ai_response,
            retrieved_chunks=retrieved_chunks
        ),
        "timestamp": datetime.now(timezone.utc)  
    }

    store_evaluation(
        query_id=query_data["_id"],
        ai_response=ai_response,
        evaluation_scores=metrics
    )
    print(f"Evaluation completed for document {query_data['_id']}\n")

def batch_evaluate(limit: int = 100):
    """Evaluate multiple recent queries with nested timestamp handling"""
    from pymongo import DESCENDING
    
    print(f"\nEvaluating {limit} most recent queries...")
    
    # Get documents with proper timestamp reference
    recent_queries = list(prompt_collection.find(
        {'user_prompt.timestamp': {'$exists': True}},  # Only docs with timestamps
        {
            '_id': 1, 
            'user_prompt.timestamp': 1, 
            'user_prompt.text': 1, 
            'ai_response.text': 1
        }
    ).sort([('user_prompt.timestamp', DESCENDING)]).limit(limit))
    
    if not recent_queries:
        print("No valid documents found with user_prompt.timestamp")
        return
    
    # Extract timestamps for display
    timestamps = [q['user_prompt']['timestamp'] for q in recent_queries]
    print(f"Processing documents from {min(timestamps)} to {max(timestamps)}")
    
    for i, query in enumerate(recent_queries, 1):
        try:
            doc_id = query['_id']
            doc_time = query['user_prompt']['timestamp']
            print(f"\n[{i}/{len(recent_queries)}] {doc_id} ({doc_time})")
            
            evaluate_query_response(doc_id, verify_latest=False)
            
        except Exception as e:
            print(f"Error evaluating {doc_id}: {str(e)}")
            continue

    print(f"\nBatch complete! Processed {len(recent_queries)} documents")

if __name__ == "__main__":
    fire.Fire({
        "single": evaluate_query_response,  # python pipeline.py single
        "batch": batch_evaluate            # python pipeline.py batch --limit 50
    })