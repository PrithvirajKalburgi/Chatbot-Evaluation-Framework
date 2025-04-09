# src/evaluation/accuracy.py
from datasets import load_metric
from bert_score import score as bertscore
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the ROUGE metric
rouge = load_metric("rouge")
# Load the BLEU metric
bleu = load_metric("bleu")

def compute_accuracy(predicted: str, reference: str, predicted_vector: np.ndarray, reference_vector: np.ndarray) -> dict:
  
    # BLEU score
    bleu_score = bleu.compute(predictions=[predicted], references=[[reference]])["bleu"]

    # ROUGE score
    rouge_score = rouge.compute(predictions=[predicted], references=[reference])

    # BERTScore
    P, R, F1 = bertscore(predicted, reference, lang="en")
    bertscore_f1 = F1.mean().item()  # Take the mean F1 score

    # Cosine Similarity
    cosine_sim = cosine_similarity([predicted_vector], [reference_vector])[0][0]

    return {
        "bleu": bleu_score,
        "rouge": rouge_score,
        "bertscore": bertscore_f1,
        "cosine_similarity": cosine_sim
    }
