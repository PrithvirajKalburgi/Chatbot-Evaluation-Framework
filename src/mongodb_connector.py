# src/mongodb_connector.py
from pymongo import MongoClient, DESCENDING
from config import MONGO_URI, MONGO_DB_NAME, MONGO_PROMPT_COLLECTION, MONGO_EVALUATION_COLLECTON
from datetime import datetime, timezone
import numpy as np
import json
from bson import SON, ObjectId


client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
prompt_collection = db[MONGO_PROMPT_COLLECTION]
evaluation_collection = db[MONGO_EVALUATION_COLLECTON]

#Fetch query from prompt collection
class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (ObjectId, np.integer, np.floating)):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

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
    return [
        chunk.get('text', '')
        for chunk in chunks_data.get('children', []) + chunks_data.get('parents', [])
        if 'text' in chunk
    ]

def convert_numpy_types(obj):
        if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj)
        elif isinstance (obj, np.ndarray):
            return obj.tolist()
        elif isinstance (obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(x) for x in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        return obj
    
def store_evaluation(query_id: str, ai_response: str, evaluation_scores: dict):
    """
    Store evaluation results in MongoDB.
    Args:
        query (str): The user query
        response (str): The chatbot's response
        metrics (dict): Dictionary of evaluation metrics (BLEU, ROUGE, BERTScore)
    """
    try:
        # Create document with native MongoDB types
        evaluation_document = {
            "query_id": query_id,
            "user_query": prompt_collection.find_one({"_id": query_id})['user_prompt']['text'],
            "ai_response": ai_response,
            "evaluation_scores": evaluation_scores,
            "timestamp": datetime.now(timezone.utc)
        }

        # Debug validation (optional)
        json.dumps(evaluation_document, cls=MongoJSONEncoder)

        # Actual MongoDB insertion
        evaluation_collection.insert_one(evaluation_document)

    except Exception as e:
        print(f"Failed to store evaluation: {str(e)}")
        print("Problematic document:", json.dumps(
            evaluation_document,
            cls=MongoJSONEncoder,
            indent=2,
            default=str
        ))
        raise


