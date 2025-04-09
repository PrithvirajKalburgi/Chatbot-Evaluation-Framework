from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import numpy as np

def compute_relevance(predicted_vector: np.ndarray, reference_vector: np.ndarray, similarity_threshold: float = 0.7) -> dict:
    cosine_sim = cosine_similarity([predicted_vector], [reference_vector])[0][0]
    relevance = accuracy_score([1 if cosine_sim >= similarity_threshold else 0], [1])

    return{
        "cosine_similarity": cosine_sim,
        "relevance": relevance
    }