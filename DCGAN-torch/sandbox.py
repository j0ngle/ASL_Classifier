from data import *
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from helpers import *
from torch import nn
import torch
import logging
import json
import random
from helpers import send_telegram_msg
from torchvision.models import inception_v3
from data import compute_embeddings, compute_fid

import numpy as np
from scipy.linalg import sqrtm

def compute_fid_numpy(real_embeddings, fake_embeddings):
    mu_real = real_embeddings.mean(axis=0)
    mu_fake = fake_embeddings.mean(axis=0)
    sq_norm = np.sum((mu_real - mu_fake) ** 2)

    C_r = np.cov(real_embeddings, rowvar=False)
    C_f = np.cov(fake_embeddings, rowvar=False)
    C_mean = sqrtm(C_r.dot(C_f))

    if np.iscomplexobj(C_mean):
        C_mean = C_mean.real

    trace = np.trace(C_r + C_f - 2*C_mean)

    return sq_norm + trace

inception = inception_v3(pretrained=True)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


inception.fc = Identity()
inception.dropout = Identity()
inception.eval()

x1 = torch.randn(64, 3, 299, 299)
x2 = torch.randn(64, 3, 299, 299)
real = inception(x1).detach().numpy()
fake = inception(x2).detach().numpy()

print(compute_fid_numpy(real, real))
print(compute_fid_numpy(real, fake))



# fs = compute_fid(real, real)
# fd = compute_fid(real, fake)

# print(f"Same {fs}\nDifferent: {fd}")
# print(fs)

# for i in range(len(real[0])):
#     r_i = real[0][i]
#     f_i = fake[0][i]
#     fid_same = compute_fid(r_i, r_i)
#     fid = compute_fid(r_i, f_i)
# print(fid)