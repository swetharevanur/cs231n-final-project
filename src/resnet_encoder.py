import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import VideoDataset

# using gpu
USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using device:', device)	

res18_model = models.resnet18(pretrained = True)
res18_encoder = nn.Sequential(*list(res18_model.children())[:-1])
res18_encoder = res18_encoder.to(device)
torch.save(res18_encoder, 'models/res18_encoder.pth')

res50_model = models.resnet50(pretrained = True)
res50_encoder = nn.Sequential(*list(res50_model.children())[:-1])
res50_encoder = res50_encoder.to(device)
torch.save(res50_encoder, 'models/res50_encoder.pth')

def load(fname = '../../jackson-clips'):
       print("Starting to load data.")
       batch_size = 128
       train_data = VideoDataset.VideoDataset(fname = fname, transform=[transforms.ToTensor()])
       train_loader = torch.utils.data.DataLoader(train_data, shuffle = True, \
           batch_size = batch_size, num_workers = 8, drop_last = True)
       return train_loader



for param in res18_encoder.parameters():
       param.requires_grad = False

for param in res50_encoder.parameters():
	param.requires_grad = False

train_loader = load()

for i, (images, _) in enumerate(train_loader):
       print("Processing image:", i)
       images = images.to(device = device, dtype = dtype)
       output_18 = res18_encoder(images)
       output_50 = res50_encoder(images)
       print(output_18.shape)
       print(output_50.shape)




