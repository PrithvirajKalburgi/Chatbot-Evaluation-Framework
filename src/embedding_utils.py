from transformers import CLIPProcessor, CLIPModel
import torch

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

def embed_text(text: str):
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)

        embedding = embedding.detach().numpy()

        return embedding
    
