import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from data import Img_Dataset
from data import rand_aug
from network import Model
from loss import *

datapath = 'C:/Users/jthra/Documents/data/PetImages'

print("Loading dataset...")
data = Img_Dataset(datapath)
dataloader = DataLoader(data, batch_size=32)
print("Dataset loaded!")

model = Model()
if torch.cuda.is_available():
    print("Sending model to CUDA device...")
    model.cuda()

optimizer = torch.optim.Adam(model.parameters())

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


# TRAINING LOOP
epochs = 10
for i in len(epochs):
    for batch, X in enumerate(dataloader):
        X1 = rand_aug(X)
        X2 = rand_aug(X)

        optimizer.zero_grad()
        output_X1 = model(X1)
        output_X2 = model(X2)

        sim = similarity(output_X1, output_X2)
        

