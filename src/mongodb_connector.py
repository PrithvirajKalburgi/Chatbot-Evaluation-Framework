# src/mongodb_connector.py
from pymongo import MongoClient
from config import MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION_NAME


client =MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
prompt_collection = db["prompt_collection"]
evaluation_collection = db["evaluation_collection"]

#Fetch query from prompt collection

def fetch_user_query(query_id: str) -> str:

    prompt = prompt_collection.find_one({"_id": query_id})

    if prompt:
        return prompt.get("user_prompt", "")
    else:
        return ""
    
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
        "ai_response": ai_response,
        "evaluation_scores": evaluation_scores
    }
    evaluation_collection.insert_one(evaluation_document)

