import swag 
import cv2

import numpy as np
import scipy
import csv
import sklearn.cross_validation
import pandas as pd
# from conv_autoencoder import AutoEncoder
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import VideoDataset
from ConvAE import AutoEncoder

from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from get_frames import getVehicleFrames

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
			frameLabels.append(frame[1]) # Frame[0] is framenumber, frame[1] is label
	return frameLabels

def split_data(X, y):
	np.random.seed(1234)
	num_sample = np.shape(X)[0]
	num_test = num_sample // 5
	num_train = num_sample - num_test
	# print(num_sample)
	# print(num_test)
	
	X_test = X[0:num_test]
	X_train = X[num_test:]

	y_test = y[0:num_test]
	y_train = y[num_test:]

	# split train/val (80/20)
	num_val = num_train // 5

	train_inds = np.arange(num_train)
	random.shuffle(train_inds)
	val_inds, train_inds = train_inds[0:num_val], train_inds[num_val]

	X_val = X_train[val_inds]
	y_val = y_train[val_inds]
	X_tr = X_train[train_inds]
	y_tr = y_train[train_inds]

	print(type(y_train))
	print(type(X_train))

	

	return np.array(X_tr.detach()), np.array(X_val.detach()), np.array(X_test.detach()), \
		np.array(y_tr), np.array(y_val), np.array(y_test)


class DataLoader(object):
	""" A class to load in appropriate numpy arrays
	"""

	def __init__(self):
		self.codes = [] # list of encoded images
		self.labels = []

	def prune_features(self, val_primitive_matrix, train_primitive_matrix, thresh=0.01):
		val_sum = np.sum(np.abs(val_primitive_matrix),axis=0)
		train_sum = np.sum(np.abs(train_primitive_matrix),axis=0)

		#Only select the indices that fire more than 1% for both datasets
		train_idx = np.where((train_sum >= thresh*np.shape(train_primitive_matrix)[0]))[0]
		val_idx = np.where((val_sum >= thresh*np.shape(val_primitive_matrix)[0]))[0]
		common_idx = list(set(train_idx) & set(val_idx))

		return common_idx

	def load_data(self, data_path = '../data/'):
		fname1 = 'jackson-town-square-2017-12-14.csv'

		# Load model
		modelName = 'models/autoencoder.pth'
		autoencoder = AutoEncoder(code_size = 50)
		# autoencoder.load_state_dict(torch.load(modelName))
		autoencoder = torch.load(modelName)
		# crop the frames
		vehicleFrames = getVehicleFrames(data_path + fname1)[0]
		frameNumsToLoad = vehicleFrames[0:10]

		bb = pd.read_csv(data_path + fname1, header = 0)
		bb_dims = ['xmin', 'ymin', 'xmax', 'ymax']

		fname2 = '../../jackson-clips'
		video = swag.VideoCapture(fname2)

		for frameNum in frameNumsToLoad:
			x_min, y_min, x_max, y_max = [bb.loc[bb['frame'] == frameNum][dim].tolist()[0] for dim in bb_dims]

			# read in frame of interest
			video.set(1, frameNum)
			ret, frame = video.read()
			if ret == False: break # EOF reached

			# crop frame
			frame = frame[int(y_min):int(y_max), int(x_min):int(x_max), :]

			# resize frame
			frame = VideoDataset.resize_frame(frame)

			# Use autoencoder to do 3-d to 1-d
			# ... but first convert to tensor that autoencoder accepts
			transform = [transforms.ToTensor()]
			for tform in transform:
			    frameTensor = tform(frame)
			frameTensor = frameTensor.unsqueeze_(0)
			frameTensor = frameTensor.to(device = device, dtype = dtype)
			code = autoencoder.encode(frameTensor)

			self.codes.append(code)	
			# get labels associated with each frame
			self.labels.append(bb.iloc[vehicleFrames[frameNum]][1] == 'car')

			if frameNum%10 == 0:
				print(frameNum)


		codeMatrix = torch.stack(self.codes, 0)
		labelTensor = torch.Tensor(self.labels)


		# split dataset into train, val, test
		train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
			train_ground, val_ground, test_ground = split_data(codeMatrix, labelTensor)

		#Prune Feature Space
		common_idx = self.prune_features(val_primitive_matrix, train_primitive_matrix)
		return train_primitive_matrix[:,common_idx], val_primitive_matrix[:,common_idx], test_primitive_matrix[:,common_idx], \
			np.array(train_ground), np.array(val_ground), np.array(test_ground)


