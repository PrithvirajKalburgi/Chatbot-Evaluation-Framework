from lettucedetect.models.inference import HallucinationDetector
from typing import List, Dict
import numpy as np

detector = HallucinationDetector(
    method="transformer", 
    model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"
)

def detect_hallucination(predicted_response: str, retrieved_chunks: List[str], threshold: float = 0.5) -> Dict [str, float]:
    combined_context = " ".join(retrieved_chunks)

    chunk_scores = []
    
    # Score against each chunk individually
    for chunk in retrieved_chunks:
        predictions = detector.predict(
            context=[chunk],
            question="",  # Not using Q&A mode
            answer=predicted_response,
            output_format="spans"
        )
        score = predictions[0]['confidence'] if predictions else 1.0
        chunk_scores.append(float(score))
    
    # Calculate overall score (average of chunk scores)
    overall_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 1.0
    
    return {
        "hallucination_score": overall_score,
        "is_hallucinated": overall_score < threshold,
        "chunk_scores": chunk_scores
    }