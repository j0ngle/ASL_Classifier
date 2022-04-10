from torch.optim import Adam
from torch.utils.data import DataLoader
from data import GAN_Dataset
from network import Generator, Discriminator
from network import initialize_weights
from network import BATCH_SIZE
from train_test import train
from torch import manual_seed, nn
import torch
from helpers import create_logfile
import random


#Seed for reproducibility
# thisSeed = 561
# thisSeed = random.randint(1, 10000)98
# print(f"Set seed: {thisSeed}") 
# random.seed(thisSeed)
# torch.manual_seed(thisSeed)

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
G_optim = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optim = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
epochs = 25

logdir = 'D:/School/Machine Learning Projects/Machine-Learning-Projects/DCGAN-torch/logs/'
create_logfile(logdir, 'train_log')

train(epochs, dataloader, generator, discriminator, G_optim, D_optim)