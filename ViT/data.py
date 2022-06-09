# from torchvision.datasets import Food101
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

root = 'C:/Users/jthra/OneDrive/Documents/data'
training_data = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=ToTensor())

# trainin_dl = DataLoader(training_data, batch_size=64)



def patch_image(img, patch_size):
    size = img.size()
    patches = []
    for i in range(int(size[1]/patch_size)):
        for j in range(int(size[2]/patch_size)):
            xs = i*patch_size
            xe = xs + patch_size
            ys = j*patch_size
            ye = ys + patch_size
            patch = img[:, xs:xe, ys:ye]
            patches.append(patch)

    return patches

def plot_patches(patches):
    dim = len(patches) ** 0.5

    plt.figure(figsize=(dim, dim))
    for index, patch in enumerate(patches):
        plt.subplot(dim, dim, index + 1)
        plt.imshow(patch.permute(1, 2, 0), cmap='binary')
        plt.axis("off")
    
    plt.show()


    

image = training_data[0][0]
print(len(training_data))
div = patch_image(image, 16)
plot_patches(div)

# plt.imshow(training_data[0][0].permute(1, 2, 0))
# plt.show()