from lettucedetect.models.inference import HallucinationDetector

detector = HallucinationDetector(
    method="transformer", 
    model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"
)

def detect_hallucination(predicted: str, retrieved_context: str, threshold: float = 0.5) -> dict:
    predictions = detector.predict( 
        context=[retrieved_context],
        question="",
        answer=predicted,
        output_format="spans"
    )
    
      

    if predictions:
        score = predictions[0]['confidence']
        hallucinated = score < threshold
    
    else:
        score = 1.0
        hallucinated = False
    
    return{
        "hallunication_score": score,
        "hallucinated": hallucinated
    }