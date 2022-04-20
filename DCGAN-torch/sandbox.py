from data import *
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from helpers import *
from torch import nn
import torch
import logging
import json
from helpers import send_telegram_msg
from torchvision.models import inception_v3

# def compute_fid(real_embeddings, fake_embeddings):
#     #Compute means and find Euclidean distance
#     mu_real = torch.mean(real_embeddings, 1)
#     mu_fake = torch.mean(fake_embeddings, 1)
#     sq_norm = torch.sum((mu_real - mu_fake) ** 2)

#     #Compute covariance matrices
#     C_r = torch.cov(real_embeddings)   
#     C_f = torch.cov(fake_embeddings)
#     C_mean = torch.sqrt(torch.mm(C_r, C_f))
#     trace = torch.trace(C_r + C_f - 2*C_mean)

#     return (sq_norm + trace).item()

# inception = inception_v3(pretrained=True)

# class Identity(nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()

#     def forward(self, x):
#         return x


# inception.fc = Identity()
# inception.dropout = Identity()
# inception.eval()

# x1 = torch.randn(64, 3, 299, 299)
# x2 = torch.randn(64, 3, 299, 299)
# real = inception(x1)
# fake = inception(x2)

# print(real)

# for i in range(len(real[0])):
#     r_i = real[0][i]
#     f_i = fake[0][i]
#     fid = compute_fid(r_i, f_i)
# print(fid)

telepath = "C:/Users/jthra/OneDrive/Documents/data/telegram.json"
f = open(telepath)
data = json.load(f)
print(data["id"])
print(data["token"])