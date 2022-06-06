import torchvision.transforms as T
from torch.utils.data import Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt

IMG_SIZE = 256

preprocess = T.Compose([
    T.Resize(IMG_SIZE),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize((.5, .5, .5), (.5, .5, .5))
])

rand_aug = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.2),
    T.RandomApply(transforms=[T.ColorJitter(brightness=.5, hue=.8, saturation=.3)]),
    T.RandomRotation(degrees=(-45, 45)),
    T.RandomInvert(p=.1)
])

def process_images(path, num_per_class=1000):
    folders = [os.path.join(path, 'Cat'), os.path.join(path, 'Dog')]
    processed = []

    for folder in folders:
        i = 0
        for filename in os.listdir(folder):
            if i >= num_per_class: break

            loc = os.path.join(folder, filename)
            img = Image.open(loc)

            try:
                img = preprocess(img)
            except Exception as e:
                #Something is wrong with image loading and
                #I can't figure out what. I'll fix it later
                continue

            processed.append(img)

            i += 1

    return processed, len(processed)
    
class Img_Dataset(Dataset):
    def __init__(self, path):
        f, l = process_images(path, num_per_class=2000)
        self.features = f
        self.length = l

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx]