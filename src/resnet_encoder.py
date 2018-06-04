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
# from conv_autoencoder import load

# using gpu
USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)	

IMAGE_WIDTH = 224 # 1920
IMAGE_HEIGHT = 224 # 1080
NUM_CHANNELS = 3
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS

def load(fname = '../../jackson-clips'):
	print("Starting to load data.")
	batch_size = 128
	train_data = VideoDataset.VideoDataset(fname = fname, transform=[transforms.ToTensor()])
	train_loader = torch.utils.data.DataLoader(train_data, shuffle = True, \
	    batch_size = batch_size, num_workers = 8, drop_last = True)
	return train_loader

res18_model = models.resnet18(pretrained=True)
res18_conv = nn.Sequential(*list(res18_model.children())[:-1])

res18_conv = res18_conv.to(device)

for param in res18_conv.parameters():
        param.requires_grad = False

train_loader = load()

for i, (images, _) in enumerate(train_loader):
    print(i)
    #inputs = Variable(inputs)
    outputs = res18_conv(images)
    print(outputs.shape)




