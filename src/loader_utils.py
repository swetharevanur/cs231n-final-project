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

def preprocess_bb(trunc_margin, bb_fname = '../data/jackson-town-square-2017-12-14.csv'):
	bb_raw = pd.read_csv(bb_fname, header = 0)
	# remove objects that only appear in one frame
	bb = bb_raw[bb_raw.groupby('ind').ind.transform(len) >= trunc_margin]
	return bb


def preprocess_frame(bb, frameNum, frame):
	bb_dims = ['xmin', 'ymin', 'xmax', 'ymax']

	# crop frame
	x_min, y_min, x_max, y_max = [bb.loc[bb['frame'] == frameNum][dim].tolist()[0] for dim in bb_dims]
	frame = frame[int(y_min):int(y_max), int(x_min):int(x_max), :]
	# resize frame
	frame = VideoDataset.resize_frame(frame)
	return frame


def frame2tensor(frame):
	# set up tensor to store encoded frames
	transform = [transforms.ToTensor()]
	for tform in transform: # convert to tensor
		frameTensor = tform(frame)
		frameTensor = frameTensor.unsqueeze_(0)
		frameTensor = frameTensor.to(device = device, dtype = dtype)
	return frameTensor


def init_encoder(mode):
	# load model
	if mode == 'auto':
		# model_fname = 'models/autoencoder_0.0001.pth'
		model_fname = 'models/epochs/autoencoder_new_epoch_optimized.pth'
		encoder = AutoEncoder()
		encoder = torch.load(model_fname)
		for param in encoder.parameters():
			param.requires_grad = False
	elif mode == 'res18':
		encoder = models.resnet18(pretrained = True)
		encoder = nn.Sequential(*list(encoder.children())[:-1])
		encoder = encoder.to(device)
		# turn off intermediate state saving
		for param in encoder.parameters():
			param.requires_grad = False
	elif mode == 'res50':
		encoder = models.resnet50(pretrained = True)
		encoder = nn.Sequential(*list(encoder.children())[:-1])
		encoder = encoder.to(device)
		# turn off intermediate state saving
		for param in encoder.parameters():
			param.requires_grad = False
	else:
		raise Exception("Illegal parameter for mode")
	return encoder


def encode(encoder, frameTensor, mode):
	if mode == 'auto':
		# autoencoder
		code = encoder.encode(frameTensor)
	elif mode == 'res18':
		# resnet_18
		code = encoder(frameTensor).squeeze(3).squeeze(2)
	elif mode == 'res50':
		# This is a resnet_50
		code_50 = encoder(frameTensor).squeeze(3).squeeze(2)
		code_50 = code_50.view(1, 512, 4)	
		code = code_50.max(dim = 2, keepdim = False)[0]
	else:
		raise Exception("Illegal parameter for mode but how did we even get here?")

	return code

