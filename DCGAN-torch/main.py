import random
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import GAN_Dataset
from network import Generator, Discriminator
from network import initialize_weights
from network import BATCH_SIZE, LR, BETAS
from train_test import train
from helpers import create_logfile

#Seed for reproducibility
thisSeed = 561
# thisSeed = random.randint(1, 10000)
print(f"Set seed: {thisSeed}") 
random.seed(thisSeed)
torch.manual_seed(thisSeed)

path = "C:/Users/jthra/OneDrive/Documents/data/img_align_celeba"

print("Loading dataset...")
data = GAN_Dataset(d_size=64, path=path)
dataloader = DataLoader(data, batch_size=BATCH_SIZE)
print("Dataset loaded!")

generator = Generator()
discriminator = Discriminator()
if torch.cuda.is_available():
    print("Sending models to cuda device...")
    generator.cuda()
    discriminator.cuda()

generator.apply(initialize_weights)
discriminator.apply(initialize_weights)


criterion = nn.BCELoss()
G_optim = Adam(generator.parameters(), lr=LR, betas=BETAS)
D_optim = Adam(discriminator.parameters(), lr=LR, betas=BETAS)
epochs = 20

logdir = 'D:/School/Machine Learning Projects/Machine-Learning-Projects/DCGAN-torch/logs/'
create_logfile(logdir, 'info')

train(epochs, dataloader, generator, discriminator, G_optim, D_optim)