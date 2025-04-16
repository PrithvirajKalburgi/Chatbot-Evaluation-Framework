from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_text(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)

    return embeddings.numpy().flatten()
    
