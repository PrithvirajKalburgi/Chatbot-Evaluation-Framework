#Importing modules

from qdrant_client import QdrantClient #QdrantClient is imported from qdrant_client library which is used to interact with a Qdrant vector database.
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION #Configuration values defined in the config.py file  

def get_qdrant_data():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT) #QdrantClient object is created using the host and port from the config file, establishes connection to Qdrant server
    results, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        with_payload=True,
        limit=1000
    )
    #scroll() : method is used to retrieve data from a collection in Qdrant
    # with_playload = True : ensures that the result includes not just the vectors, but also any additional metadata (payload). 
    # limit=1000 : limits the number of results to 1000
    # the method returns a tuple: a list of results and a "next page" offset (not used here, hence the "_")
    return results