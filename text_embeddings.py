import open_clip
import torch
import pandas as pd
import os
from config import *
from tqdm import tqdm

def save_embeddings():
    return

def get_embedding_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(embedding_model, pretrained=embedding_pretrained, device=device)
    model = model.to(device)
    model.eval()
    return model, device

def get_text_embedding(model, caption, device):
    tokenizer = open_clip.get_tokenizer(embedding_model)
    tokens = tokenizer(caption).to(device) # (1, 1024)
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding # (1, 1024)
    
if __name__ == "__main__":
    save_embeddings()