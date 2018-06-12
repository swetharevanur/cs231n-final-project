import swag 
import cv2
import numpy as np
import scipy
import csv
import sklearn.cross_validation
import pandas as pd
import random
from random import shuffle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

import VideoDataset
from get_frames import getVehicleFrames

USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Test
path = './models/'

# goodae = 'autoencoder_fully_trained.pth'
oldae = 'autoencoder.pth'
res18 = 'res18_encoder.pth'
res50 = 'res50_encoder.pth'
'''
goodae = 'models/epochs/autoencoder_new_epoch_optimized.pth'
encoder = AutoEncoder()
encoder = torch.load(model_fname)
for param in encoder.parameters():
	param.requires_grad = False
'''
encoder = models.resnet18(pretrained = True)
encoder = nn.Sequential(*list(encoder.children())[:-1])
encoder = encoder.to(device)
# turn off intermediate state saving
for param in encoder.parameters():
	param.requires_grad = False


print(encoder)
