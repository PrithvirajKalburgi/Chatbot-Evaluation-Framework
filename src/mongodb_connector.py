# src/mongodb_connector.py
from pymongo import MongoClient
from pymongo import DESCENDING
from config import MONGO_URI, MONGO_DB_NAME, MONGO_PROMPT_COLLECTION, MONGO_EVALUATION_COLLECTON
from datetime import datetime, timezone


client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
prompt_collection = db[MONGO_PROMPT_COLLECTION]
evaluation_collection = db[MONGO_EVALUATION_COLLECTON]

#Fetch query from prompt collection

def fetch_user_query(query_id=None): 

    if query_id:
       return prompt_collection.find_one({"_id": query_id})
    else:
        return prompt_collection.find().sort('timestamp', DESCENDING).limit(1)[0]
    
def fetch_retrieved_chunks(query_id: str):
    doc = prompt_collection.find_one({"_id": query_id})
    if not doc or 'retrieved_context' not in doc:
        return []
    
    chunks_data = doc['retrieved_context']['chunks']
    children = chunks_data.get('children', [])
    parents = chunks_data.get('parents', [])

    all_chunks = children + parents
    return [chunk.get('text', '') for chunk in all_chunks if 'text' in chunk]
    
def store_evaluation(query_id: str, ai_response: str, evaluation_scores: dict):
    """
    Store evaluation results in MongoDB.
    Args:
        query (str): The user query
        response (str): The chatbot's response
        metrics (dict): Dictionary of evaluation metrics (BLEU, ROUGE, BERTScore)
    """
    evaluation_document = {
        "query_id": query_id,
        "user_query": prompt_collection.find_one({"_id": query_id})['user_prompt']['text'],
        "ai_response": ai_response,
        "evaluation_scores": evaluation_scores,
        "timestamp": datetime.now(timezone.utc)
    }
    evaluation_collection.insert_one(evaluation_document)

