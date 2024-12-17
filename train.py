import time

import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transfroms

import models

ds_transfrom = transfroms.Compose([
    transfroms.ToImage(),
    transfroms.ToDtype(torch.float32,scale=True)
])
ds_train = datasets.FashionMNIST()