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

for i in range(0, 3):
    print(i)