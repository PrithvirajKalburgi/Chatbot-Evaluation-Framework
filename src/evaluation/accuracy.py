# src/evaluation/accuracy.py
from evaluate import load
from bert_score import score as bertscore
from sklearn.metrics.pairwise import cosine_similarity
from embedding_utils import embed_text
from typing import List
import numpy as np

# Load the ROUGE metric
rouge = load("rouge")
# Load the BLEU metric
bleu = load("bleu")

def compute_accuracy(predicted: str, reference_chunks: str, predicted_embedding: np.ndarray, reference_embeddings: List[np.ndarray]) -> dict:

    predicted_embedding = predicted_embedding.flatten()
    reference_embedding = [embed.flatten() for embed in reference_embeddings]
  
    # BLEU score
    bleu_scores = [
        bleu.compute(predictions=[predicted], references=[[chunk]])["bleu"]
        for chunk in reference_chunks

    ]

    # ROUGE score
    rouge_scores = []
    for chunk in reference_chunks:
        scores = rouge.compute(predictions=[predicted], references=[chunk])
        rouge_scores.append({
            "rouge1": scores["rouge1"],
            "rouge2": scores["rouge2"],
            "rougeL": scores["rougeL"]
        })

    # BERTScore
    combined_reference = " ".join(reference_chunks)
    _, _, bert_f1 = bertscore([predicted], [combined_reference], lang="en")
    
    # Cosine Similarity
    cosine_sims = [
        cosine_similarity([predicted_embedding], [chunk_embed])[0][0]
        for chunk_embed in reference_embeddings
    ]

    return {
        "bleu": {
            "min": float(min(bleu_scores)),
            "max": float(max(bleu_scores)),
            "mean": float(sum(bleu_scores) / len(bleu_scores))
        },
        "rouge": {
            "rouge1": float(average_metric(rouge_scores, "rouge1")),
            "rouge2": float(average_metric(rouge_scores, "rouge2")), 
            "rougeL": float(average_metric(rouge_scores, "rougeL"))
        },
        "bertscore": float(bert_f1.mean().item()),
        "cosine_similarity": {
            "min": float(min(cosine_sims)),
            "max": float(max(cosine_sims)),
            "mean": float(sum(cosine_sims) / len(cosine_sims))
        }
    }

def average_metric (scores: List[dict], metric_name: str) -> float:
    return sum(s[metric_name] for s in scores) / len(scores)
