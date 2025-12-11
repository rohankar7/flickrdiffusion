import numpy as np
import pandas as pd
import os
from config import *
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as T

class ImageDataset(Dataset):
    def __init__(self, file_names):
        self.file_names = file_names
        self.transform = T.Compose([
            # T.Resize((image_res, image_res)),
            T.ToTensor(),  # -> CxHxW, values in [0,1]
        ])
    def __len__(self): return len(self.file_names)
    def __getitem__(self, index):
        file_path = self.file_names[index]
        image_data = Image.open(file_path).convert("RGB")
        # assert image_data.shape == torch.Size([1, image_res, image_res]), f"Unexpected shape: {image_data.shape}"
        return self.transform(image_data)
def image_dataloader():
    image_paths = [os.path.join(resized_img_dir, path) for path in os.listdir(resized_img_dir)[:] if path.endswith(".jpg")]
    dataset = ImageDataset(image_paths)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=vae_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=vae_batch_size, shuffle=False)
    return train_loader, test_loader

class Flickr8kImageCaption(Dataset):
    def __init__(self, file_names):
        self.file_names = file_names
    def __len__(self): return len(self.file_names)
    def __getitem__(self, index):
        file_path = self.file_names[index]
        img_data = torch.load(f"{image_dir}/{file_path}", map_location="cpu", weights_only=False)
        embedding_data = torch.load(f"{embedding_dir}/{file_path}", map_location="cpu", weights_only=False)
        # assert img_data.shape == 
        assert embedding_data.shape == torch.Size([1, 1024]), f"Unexpected embedding shape: {embedding_data.shape}"
        return (img_data, embedding_data)
def img_emb_dataloader():
    file_names = [path for path in os.listdir(image_dir) if path.endswith(".pt")]
    dataset = Flickr8kImageCaption(file_names)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=unet_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=unet_batch_size, shuffle=False)
    return train_loader, test_loader