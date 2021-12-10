import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np

preprocess = T.Compose([
    T.Resize(32),
    T.CenterCrop(32),
    T.Normalize(
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5]
    )
])



def process_images(path, d_size):
    processed = []
    i = 0

    for filename in os.listdir(path):
        if filename.endswith('jpg'):
            if (i % 500 == 0):
                print("[PREPROCESSING] Processed {} images".format(i))

        img = torchvision.io.read_image(filename)
        processed.append(preprocess(img))

        i+=1

    return processed

class GAN_Dataset(Dataset):
    def __init__(self, d_size, path):
        f, l = process_images(path, d_size)
        self.features = f
        self.length = l

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx]

