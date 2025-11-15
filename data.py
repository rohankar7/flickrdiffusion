import numpy as np
import pandas as pd
import os
import config
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class Flickr8kImageCaption(Dataset):
    def __init__(self):
        self.