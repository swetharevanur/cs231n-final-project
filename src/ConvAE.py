import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import VideoDataset

IMAGE_WIDTH = 224 # 1920
IMAGE_HEIGHT = 224 # 1080
NUM_CHANNELS = 3
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS

code_size = 512

class Flatten(nn.Module):
	def forward(self, x):
		N, C, H, W = x.size()
		return x.view(N, -1)

class Unflatten(nn.Module):
	def __init__(self, N=-1, C=20, H=28, W=28):
		super(Unflatten, self).__init__() 
		self.N = N
		self.C = C
		self.H = H
		self.W = W
	def forward(self, x):
		return x.view(self.N, self.C, self.H, self.W)

class AutoEncoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.code_size = code_size
		
		# Encoder specification
		self.enc_cnn_1 = nn.Conv2d(NUM_CHANNELS, 20, kernel_size=3, padding=1)
		self.enc_cnn_2 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
		self.enc_cnn_3 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
		self.enc_linear_1 = nn.Linear(28 * 28 * 20, self.code_size)
		
		# Decoder specification
		self.dec_linear_1 = nn.Linear(self.code_size, 28 * 28 * 20)
		self.batch_1 = nn.BatchNorm1d(28 * 28 * 20)
		self.dec_cnn_1 = nn.ConvTranspose2d(20, 20, 4, stride = 2, padding = 1) # Now 56 from 28
		self.batch_2 = nn.BatchNorm2d(20)
		self.dec_cnn_2 = nn.ConvTranspose2d(20, 20, 4, stride = 2, padding = 1) # Now 112 from 56
		self.batch_3 = nn.BatchNorm2d(20)
		self.dec_cnn_3 = nn.ConvTranspose2d(20, 3, 4, stride = 2, padding = 1) # Now 224 from 112, 3 layers now

		# Not using Sequential API rn
		self.encoder = nn.Sequential(
			nn.ReLU(nn.Conv2d(NUM_CHANNELS, 20, kernel_size=3, padding=1)),
			nn.ReLU(nn.Conv2d(20, 20, kernel_size=3, padding=1)),
			nn.ReLU(nn.Conv2d(20, 20, kernel_size=3, padding=1)),
			Flatten(),
			nn.Linear(28 * 28 * 20, 512)
		)

		# Not using Sequential API rn
		self.decoder = nn.Sequential(
			nn.ReLU(nn.Linear(self.code_size, 28 * 28 * 20)),
			nn.BatchNorm1d(28 * 28 * 20),
			Unflatten(),
			nn.ReLU(nn.ConvTranspose2d(20, 20, 4, stride = 2, padding = 1)),
			nn.BatchNorm2d(20),
			nn.ReLU(nn.ConvTranspose2d(20, 20, 4, stride = 2, padding = 1)),
			nn.BatchNorm2d(20),
			nn.ConvTranspose2d(20, 3, 4, stride = 2, padding = 1),
			# nn.Tanh()
			# WOULD NEED A SIGMOID HERE
		)
		
	def forward(self, images):
		# image size is now 224 x 224 (formerly 1920 x 1080)
		code = self.encode(images)
		out = self.decode(code)
		return out, code
	
	def encode(self, images):
		code = self.enc_cnn_1(images)
		code = F.relu(F.max_pool2d(code, 2))
		code = self.enc_cnn_2(code)
		code = F.relu(F.max_pool2d(code, 2))
		code = self.enc_cnn_3(code)
		code = F.relu(F.max_pool2d(code, 2))
		code = code.view([images.size(0), -1])
		code = self.enc_linear_1(code)
		return code

	def decode(self, code):
		out = F.relu(self.dec_linear_1(code))
		out = self.batch_1(out)
		out = out.view([out.size(0), 20, 28, 28]) # N, C, H, W
		out = F.relu(self.dec_cnn_1(out))
		out = self.batch_2(out)
		out = F.relu(self.dec_cnn_2(out))
		out = self.batch_3(out)
		out = F.sigmoid(self.dec_cnn_3(out))
		return out


