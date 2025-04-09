from lettucedetect import Lettuce

lettuce = Lettuce(mode_name="lettuce-bert-base-uncased")

def detect_hallucination(predicted: str, retrieved_context: str, threshold: float = 0.5) -> dict:
    result = lettuce.score(response=predicted, context=retrieved_context)

    score = result["score"]
    hallucinated = score < threshold

    return {
        "hallucination_score": score,
        "hallucinated": hallucinated
    }