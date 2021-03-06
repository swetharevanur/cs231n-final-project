# Adapted from: https://gist.github.com/okiriza/16ec1f29f5dd7b6d822a0a3f2af39274

import random
import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import VideoDataset
from ConvAE import Flatten, Unflatten, AutoEncoder
import os

# using gpu
USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using device:', device)	

IMAGE_WIDTH = 224 # 1920
IMAGE_HEIGHT = 224 # 1080
NUM_CHANNELS = 3
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS

results_fname = 'models/results/autoencoder_new_epoch_exps.txt'
# os.remove(results_fname)

# load data
def load(fname = '../../jackson-clips'):
	print("Starting to load data.")
	batch_size = 128
	train_data = VideoDataset.VideoDataset(fname = fname, transform=[transforms.ToTensor()])
	train_loader = torch.utils.data.DataLoader(train_data, shuffle = True, \
	    batch_size = batch_size, num_workers = 8, drop_last = True)
	return train_loader

# instantiate model
def create_model(lr, cont = False):
	autoencoder = AutoEncoder()
	if cont == True:
		autoencoder = torch.load('models/epochs/autoencoder_new_epoch_optimized.pth')
	autoencoder = autoencoder.to(device = device)
	loss_fn = nn.BCELoss()
	optimizer_cls = optim.Adam
	optimizer = optimizer_cls(autoencoder.parameters(), lr = lr)
	return autoencoder, loss_fn, optimizer

# training model
def train_model(autoencoder, loss_fn, optimizer, num_epochs, save, running_lowest_loss):
	for epoch in range(num_epochs):
		print("Epoch %d" % epoch)
		for i, (images, _) in enumerate(train_loader): # ignore image labels
			images = images.to(device = device, dtype = dtype)
			out, code = autoencoder(images)
			optimizer.zero_grad()
			loss = loss_fn(out, images)
			loss.backward()
			optimizer.step()
		print("Epoch Loss = %.5f" % loss.data[0])
		with open(results_fname, 'a') as text_file:
			text_file.write("\nEpoch = %d" % epoch)
			text_file.write("\nEpoch Loss = %.5f" % loss.data[0])

		# save intermediate models
		if save:
			if loss.data[0] < running_lowest_loss:
				running_lowest_loss = loss
				torch.save(autoencoder, 'models/epochs/autoencoder_new_epoch_optimized.pth')

	print("Final Loss = %.5f" % loss.data[0])	
	return loss.data[0]

# hyperparameter sweep
def tune(num_epochs_arr, lr_arr):
	running_lowest_loss = math.inf
	running_best_model = None
	running_best_params = None

	for num_epochs in num_epochs_arr:
		for lr in lr_arr:
			autoencoder, loss_fn, optimizer = create_model(lr, cont = False)

			# print('\nLearning Rate:', lr)
			loss = train_model(autoencoder, loss_fn, optimizer, num_epochs, save = True, running_lowest_loss=running_lowest_loss)
			# write to a file too
			# with open(results_fname, 'a') as text_file:
			# 	text_file.write("\nLearning Rate = %.5f" % lr)
			# 	text_file.write("\nLoss = %.5f" % loss)

			# if loss < running_lowest_loss:
			#	running_lowest_loss = loss
			#	running_best_model = autoencoder
			#	running_best_params = {'num_epochs': num_epochs, 'lr': lr}
			
			torch.save(autoencoder, 'models/autoencoder_new_fully_trained.pth')


			del autoencoder

	# print("\nBest Model Loss = %.5f" % running_lowest_loss)
	# print("\nModel Parameters: ", running_best_params)
	# write to a file too
	# with open(results_fname, 'a') as text_file:
	#	text_file.write("\n\nBest Model Loss = %.5f" % running_lowest_loss)
	#	text_file.write("\nModel Parameters: %.5f" % running_best_params)

	# torch.save(running_best_model, 'models/autoencoder_fully_trained.pth')


# hyperparameters
num_epochs_arr = [30]
lr_arr = [1e-4]
# lr_arr = [5e-4, 1e-4, 1e-3, 5e-5, 1e-5]

with open(results_fname, 'a') as text_file:
	text_file.write("Tracking epoch loss for autoencoder with 1e-4 learning rate\n")
	# text_file.write(str(lr_arr))

train_loader = load()
tune(num_epochs_arr, lr_arr)
