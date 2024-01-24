import random
import torch
import logging
import json
import time
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import GAN_Dataset
from data import compute_fid_numpy
from network import Generator, Discriminator
from network import initialize_weights
from network import BATCH_SIZE, LR, BETAS, LATENT
from train_test import train, train_step
from helpers import *

print("START")

SEND_TELEGRAM = False

datapath = "C:/Users/jthra/OneDrive/Documents/data/img_align_celeba"
# datapath = "D:/School/landscape"
telepath = "C:/Users/jthra/OneDrive/Documents/data/telegram.json"
logpath  = "D:/West Virginia University/Machine Learning Projects/Machine-Learning-Projects/DCGAN-torch/logs/"

f = open(telepath)
telegram = json.load(f)
id = telegram['id']
token = telegram['token']
f.close()

create_logfile(logpath, 'info')

#Seed for reproducibility
thisSeed = 561
# thisSeed = random.randint(1, 10000)
print(f"Set seed: {thisSeed}") 
random.seed(thisSeed)
torch.manual_seed(thisSeed)

print("Loading dataset...")
data = GAN_Dataset(d_size=64, path=datapath)
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

G_params = count_parameters(generator)
D_params = count_parameters(discriminator)
s = f'Total trainable parameters: {G_params + D_params}'
print(s)
logging.info(s)


# criterion = nn.BCELoss()
G_optim = Adam(generator.parameters(), lr=LR, betas=BETAS)
D_optim = Adam(discriminator.parameters(), lr=LR, betas=BETAS)
epochs = 50

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


###############
#TRAINING LOOP#
###############
print("EHRE")

fixed_real = next(iter(dataloader))[0:64] #This was 64

fixed_noise = torch.randn(64, LATENT, 1, 1, device=device)
# fixed_noise = torch.randn(size=[BATCH_SIZE, LATENT], device=device)


size = len(dataloader)
generator.train()
discriminator.train()

fid_list = []
G_losses = []
D_losses = []
out      = ''

for epoch in range(epochs):

    start = time.time()
    print("\n--\n")
    for batch, X in enumerate(dataloader):
        Gl, Dl, D_x, D_G_z= train_step(X, generator, discriminator, G_optim, D_optim, device, d_pretrain=1)

        G_losses.append(Gl.item())
        D_losses.append(Dl.item())

        if batch % 50 == 0:
            out = f"[{epoch+1:d}/{epochs:d}][{batch:d}/{size:d}]\tLoss_D: {Dl.item():.4f}\tLoss_G: {Gl.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z[0]:.4f}, {D_G_z[1]:.4f}"

            print(out)
            logging.info(out)

    save_graph("G_D_loss", "Iterations", "Loss", epoch, G_losses, "G", D_losses, "D")
    
    with torch.no_grad():
        test_batch = generator(fixed_noise).detach().cpu()
    save_images(test_batch, epoch, n_cols=8)

    if (epoch + 1) % 10 == 0:
        torch.save(generator, f'models\\model_e{epoch+1}.pt')

    # print("[UPDATE] Computing FID score...")
    # fid = compute_fid_numpy(fixed_real, test_batch)
    # fid_list.append(fid)
    # logging.info(f"[UPDATE] FID at epoch {epoch+1}/{epochs}: {fid}")
    # save_graph("FID per epoch", "Epoch", "Score", epoch, fid_list, 'fid')
    # print(f"[UPDATE] FID Computed: {fid}")

    if SEND_TELEGRAM:
        out = f"[{epoch+1}/{epochs}]\nAvg loss D: {np.mean(D_losses)}\nAvg loss G: {np.mean(G_losses)}\nFID: {fid}"
        send_telegram_msg(out, id, token)

    end = time.time()
    print(f"Epoch {epoch+1} time: {end-start}")

# train(epochs, dataloader, generator, discriminator, G_optim, D_optim)