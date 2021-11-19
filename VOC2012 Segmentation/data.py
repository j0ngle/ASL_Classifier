import os
import torch
import torchvision
from torch.utils.data import Dataset
from d2l import torch as d2l
import numpy as np
import matplotlib.pyplot as plt

#Preprocessing pipeline from: d2l.ai, chapter 13

#Label maps
COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']



def download_data():
    d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                            '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

    voc_dir = d2l.download_extract('voc2012', 'data/VOCdevkit/VOC2012')

def read_images(voc_dir, is_train=True):
    '''
    The training, validation, and test sets are divided based on image labels
    in the .txt files found in /ImageSet/Segmentation.

    This function parses the desired file and creates lists of feature and label images
    that is then returned for use

    args:
        voc_dir - directory of the saved VOC2012 dataset
        is_train - Boolean specifying if you are creating training dataset
    '''

    txt_filename = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 
                                'train.txt' if is_train else 'val_text')
    mode = torchvision.io.image.ImageReadMode.RGB

    with open(txt_filename, 'r') as file:
        images = file.read().split()

    features = []
    labels = []

    for i, filename in enumerate(images):
        features.append(torchvision.io.read_image(
            os.path.join(voc_dir, 'JPEGImages', f'{filename}.jpg')
        ))

        labels.append(torchvision.io.read_image(
            os.path.join(voc_dir, 'SegmentationClass', f'{filename}.png'), 
            mode
        ))

    return features, labels

def cmap_to_label():
    '''
    Maps the indexes of CLASSES to the colormap of the label image.

    For example, all instances of [128, 0, 0] in a label will be mapped
    to a 1 because that is the corresponding index of the proper label, "aeroplane" 
    '''

    #Ngl I'm really not following along with the math in this code
    cmap2label = torch.zeros(256**3, dtype=torch.long)

    for i, cmap in enumerate(COLORMAP):
        cmap2label[(cmap[0] * 256 + cmap[1]) * 256 + cmap[2]] = i

    return cmap2label

def label_indices(cmap, cmap2label):
    """Map any RGB values in VOC labels to their class indices."""

    #Again, not really following along here
    colormap = cmap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 +
           colormap[:, :, 2])
    return cmap2label[idx]

class VOC_Dataset(Dataset):
    def __init__(self, is_train, crop_size, dir):
        f, l = read_images(dir, is_train=is_train)
        self.features = f
        self.labels = l
        self.c2l = cmap_to_label()
        self.crop_size = crop_size

        #TODO: Filter images that are smaller than crop_size
        #TODO: Normalize images
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        #TODO: Crop images and return

        return 0

