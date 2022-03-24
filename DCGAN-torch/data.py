import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from network import IMG_SIZE

preprocess = T.Compose([
    T.Resize(IMG_SIZE),
    T.CenterCrop(IMG_SIZE)
])

def process_images(path, d_size):
    processed = []
    i = 0
    to_tensor = T.ToTensor()

    for filename in os.listdir(path):
        if i >= 5000:
            break

        if filename.endswith('jpg'):
            if (i % 500 == 0):
                print("[PREPROCESSING] Processed {} images".format(i))

        loc = os.path.join(path, filename)
        img = Image.open(loc)
        img = to_tensor(img)
        processed.append(preprocess(img))

        i+=1

    return processed, len(processed)


class GAN_Dataset(Dataset):
    def __init__(self, d_size, path):
        f, l = process_images(path, d_size)
        self.features = f
        self.length = l

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx]

