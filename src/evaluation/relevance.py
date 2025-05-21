from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict

def compute_relevance(query_embedding: np.ndarray, response_embedding: np.ndarray, similarity_threshold: float = 0.5) -> Dict[str, float]:

    query_embed = query_embedding.flatten()
    response_embed = response_embedding.flatten()

    cosine_sim = cosine_similarity(
        [query_embed], [response_embed]
        )[0][0]
    
    return{
        "cosine_similarity": float(cosine_sim),
        "is_relevant":  bool(cosine_sim >= similarity_threshold) 
    }




