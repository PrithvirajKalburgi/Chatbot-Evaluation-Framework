from lettucedetect.models.inference import HallucinationDetector
from typing import List, Dict

detector = HallucinationDetector(
    method="transformer", 
    model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"
)

def detect_hallucination(predicted_response: str, retrieved_chunks: List[str], threshold: float = 0.5) -> Dict [str, float]:
    combined_context = " ".join(retrieved_chunks)

    predictions = detector.predict( 
        context=[combined_context],
        question="",
        answer=predicted_response,
        output_format="spans"
    )
     
    if predictions:
        score = predictions[0]['confidence']
        return {
            "hallucination_score": float(score),
            "is_hallucinated": bool(score < threshold),
            "chunk_score": [score]
        }
        
    else:    
        return{
            "hallunication_score": 1.0,
            "is_hallucinated": False,
            "chunk_scores": []
    }