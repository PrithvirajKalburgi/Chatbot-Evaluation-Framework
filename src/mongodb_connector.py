# Importing modules
from pymongo import MongoClient # MongoClient imported from pymongo library, used to connect to a MongoDB database.
from config import MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION_NAME #Configuration values mentioned in the config file

def get_collection():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    return db[MONGO_COLLECTION_NAME]

    # This function connects to MongoDB using the URI and accesses a specific database (MONGO_DB_NAME) and returns the desired collection (MONGO_COLLECTION_NAME) from that database.

def insert_result(result: dict):
    collection = get_collection()
    collection.insert_one(result)

    #Gets the MongoDB collection using get_collection(). Inserts one document (Python dictionary) into the collection using insert_one(). 
    #Result parameter is expected to be a dictionary (dict) that can be stored in MongoDB. 