from mongodb_connector import fetch_user_query, store_evaluation
from qdrant_connector import fetch_relevant_data
from evaluation.accuracy import compute_accuracy
from evaluation.relevance import compute_relevance
from evaluation.hallucination import detect_hallucination

def evaluate_query(query_id: str):
    # Fetch the user query from MongoDB using its ID
    user_query_data = fetch_user_query(query_id)
    user_query = user_query_data['user_prompt']  # Assuming user_prompt contains the actual query text
    ai_response = user_query_data['ai_response']  # Assuming ai_response contains the chatbot's response
    
    # Fetch relevant data from Qdrant for the given query
    retrieved_data = fetch_relevant_data(user_query)  # Assuming this returns relevant chunks of data from Qdrant

    # Assuming retrieved_data contains the relevant data and its corresponding embeddings
    relevant_texts = [data['text'] for data in retrieved_data]  # Extracting text from the retrieved data
    relevant_embeddings = [data['embedding'] for data in retrieved_data]  # Extracting embeddings from the retrieved data

    # Now, we can compute the evaluation metrics
    # For simplicity, let's assume the first relevant text is the reference (you can adjust based on your needs)
    reference = relevant_texts[0]
    reference_embedding = relevant_embeddings[0]

    # Evaluate accuracy using the reference data and the AI response
    accuracy_metrics = compute_accuracy(ai_response, reference, ai_response, reference)  # Replace with actual embeddings if available
    
    # Evaluate relevance based on cosine similarity threshold (e.g., 0.7)
    relevance_metrics = compute_relevance(ai_response, reference, ai_response, reference, similarity_threshold=0.7)
    
    # Detect hallucinations in the AI response
    hallucination_detection = detect_hallucination(ai_response, reference)

    # Store the evaluation results in MongoDB
    evaluation_scores = {
        "accuracy": accuracy_metrics,
        "relevance": relevance_metrics,
        "hallucination": hallucination_detection
    }
    
    store_evaluation(query_id, ai_response, evaluation_scores)

# Example usage: evaluate the query with a specific query ID
query_id = "your_query_id_here"
evaluate_query(query_id)