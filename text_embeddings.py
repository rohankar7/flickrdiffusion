import open_clip
import torch
import pandas as pd
import os
from config import *
from tqdm import tqdm

def save_embeddings():
    os.makedirs(f"./{embedding_dir}", exist_ok=True)
    model, device = get_embedding_model()
    with open("./data/captions.txt", "r") as f:
        for l in tqdm(f.readlines(), desc=f"Progress"):
            img_name, caption = l.strip().split(".jpg,")
            if caption[-1]==".":
                caption = caption[:-2]
            if f"{img_name}.pt" not in os.listdir(embedding_dir):
                torch.save(get_text_embedding(model, caption, device), f"./{embedding_dir}/{img_name}.pt")
    # return

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