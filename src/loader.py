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
from ConvAE import AutoEncoder
from get_frames import getVehicleFrames
from loader_utils import *

USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def parse_file(filename):
	frameLabels = []
	with open(filename, 'rt') as f:
		framereader = csv.reader(f, delimiter = ',')
		next(framereader)
		for frame in framereader:
			frameLabels.append(frame[1]) # frame[0] is frame number, frame[1] is label
	return frameLabels

def split_data(X, y, frameNums):
	np.random.seed(1234)
	num_sample = np.shape(X)[0]

	# split total into train/test (80/20)
	num_test = num_sample // 5
	num_train = num_sample - num_test
	
	X_test = X[0:num_test]
	X_train = X[num_test:]

	y_test = y[0:num_test]
	y_train = y[num_test:]

	frameNums_test = frameNums[0:num_test]
	frameNums_train = frameNums[num_test:]

	# split train into train/val (80/20)
	num_val = num_train // 5

	train_inds = np.arange(num_train)
	random.shuffle(train_inds)
	val_inds, train_inds = train_inds[0:num_val], train_inds[num_val:]

	X_val = X_train[val_inds]
	y_val = y_train[val_inds]
	X_tr = X_train[train_inds]
	y_tr = y_train[train_inds]

	frameNums_tr = frameNums_train[train_inds]

	X_tr = torch.squeeze(X_tr, 1)
	X_val = torch.squeeze(X_val, 1)
	X_test = torch.squeeze(X_test, 1)

	return np.array(X_tr.detach()), np.array(X_val.detach()), np.array(X_test.detach()), \
		np.array(y_tr), np.array(y_val), np.array(y_test), np.array(frameNums_tr)

class DataLoader(object):
	""" A class to load in appropriate numpy arrays
	"""
	def __init__(self):
		self.codes = [] # list of encoded images
		self.labels = [] # associated labels

	def load_data(self, mode = 'auto', data_path = '../data/', numFramesToLoad = 1000, need_split = True):
		labels_fname = 'jackson-town-square-2017-12-14.csv'

		

		# load model
		# encoder = init_encoder(mode)
		if mode == 'auto':
		    model_fname = 'models/autoencoder_0.0001.pth'
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
		

		
		# get unique frames with vehicles
		vehicleFrames = getVehicleFrames(data_path + labels_fname)[0]
		shuffle(vehicleFrames)

		# bb = preprocess_bb()
		bb = pd.read_csv(data_path + labels_fname, header = 0)
		bb_dims = ['xmin', 'ymin', 'xmax', 'ymax']
		
		video_fname = '../../jackson-clips'
		video = swag.VideoCapture(video_fname)

		carCount = 0
		truckCount = 0
		margin = 5
		numFramesLoaded = 0

		frameNums = np.empty((0))
		for frameIter in range(len(vehicleFrames)):
			frameNum = vehicleFrames[frameIter] # frame number as it appears in BB dataset

			# force class balancing
			vehicleType = bb.loc[bb['frame'] == frameNum]['object_name'].to_string()
			vehicleType = vehicleType.split(' ')[-1]
			if vehicleType == 'car':
				if carCount - truckCount > margin: continue # more cars than trucks, so skip
				carCount += 1
			if vehicleType == 'truck':
				if truckCount - carCount > margin: continue # more trucks than cars, so skip
				truckCount += 1
			
			# read in frame of interest
			video.set(1, frameNum)
			ret, frame = video.read()
			if ret == False: break # EOF reached
			
			# crop and resize frame
			# frame = preprocess_frame(bb, frameNum, frame)
			# print(frame.shape)
				
			
			# crop frame
			x_min, y_min, x_max, y_max = [bb.loc[bb['frame'] == frameNum][dim].tolist()[0] for dim in bb_dims]
			# print(x_min, y_min, x_max, y_max)
			frame = frame[int(y_min):int(y_max), int(x_min):int(x_max), :]
			# print(frame.shape)
			# resize frame
			frame = VideoDataset.resize_frame(frame)
			# print(frame.shape)
			

			# use autoencoder to generate 1D code from 3D image
			# frameTensor = frame2tensor(frame)
			
			transform = [transforms.ToTensor()]
			for tform in transform: # convert to tensor
				frameTensor = tform(frame)
			frameTensor = frameTensor.unsqueeze_(0)
			frameTensor = frameTensor.to(device = device, dtype = dtype)
			

			# encode
			# code = encode(encoder, frameTensor, mode)
			if mode == 'auto':
			    # This is an autoencoder
			    code = encoder.encode(frameTensor)
			elif mode == 'res18':
			    # This is a resnet_18
			    code = encoder(frameTensor).squeeze(3).squeeze(2)
			elif mode == 'res50':
			    # This is a resnet_50
			    code_50 = encoder(frameTensor).squeeze(3).squeeze(2)
			    code_50 = code_50.view(1, 512, 4)	
			    code = code_50.max(dim = 2, keepdim = False)[0]
			else:
			    raise Exception("Illegal parameter for mode but how did we even get here?")

			# Codes should all be 512 now
			self.codes.append(code)
 
			# get labels associated with each frame
			self.labels.append((vehicleType == 'car') - (vehicleType == 'truck')) # -1 is truck, 1 is car
 
			numFramesLoaded += 1
			frameNums = np.append(frameNums, frameNum)
			# if numFramesLoaded % 20 == 0:
			print(numFramesLoaded, "frames successfully loaded out of", numFramesToLoad)	
			if numFramesLoaded >= numFramesToLoad: break
 
		# report vehicle statistics for class balancing
		print("\nCar count:", carCount)
		print("Truck count:", truckCount)
		print(carCount - truckCount, "more cars than trucks.") 
 
		# convert to encoded images and labels to tensors
		codeMatrix = torch.stack(self.codes, 0)
		labelTensor = torch.Tensor(self.labels)
 
		if need_split == True:
			# split dataset into train, val, test
			train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
				train_ground, val_ground, test_ground, frameNums_train = split_data(codeMatrix, labelTensor, frameNums)

			return train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
				np.array(train_ground), np.array(val_ground), np.array(test_ground), mode, frameNums_train
		else:
			return _, codeMatrix, _, \
				_, np.array(labelTensor), _, mode, frameNums

