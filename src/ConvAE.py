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

code_size = 100

class AutoEncoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.code_size = code_size
		
		# Encoder specification
		self.enc_cnn_1 = nn.Conv2d(NUM_CHANNELS, 10, kernel_size=5)
		self.enc_cnn_2 = nn.Conv2d(10, 20, kernel_size=5)
		self.enc_linear_1 = nn.Linear(53 * 53 * 20, self.code_size*2)
		self.enc_linear_2 = nn.Linear(self.code_size*2, self.code_size)
		
		# Decoder specification
		self.dec_linear_1 = nn.Linear(self.code_size, 160)
		self.dec_linear_2 = nn.Linear(160, IMAGE_SIZE)
		
	def forward(self, images):
		# image size is now 224 x 224 (formerly 1920 x 1080)
		code = self.encode(images)
		out = self.decode(code)
		return out, code
	
	def encode(self, images):
		code = self.enc_cnn_1(images)
		code = F.selu(F.max_pool2d(code, 2))
		
		code = self.enc_cnn_2(code)
		code = F.selu(F.max_pool2d(code, 2))
		
		code = code.view([images.size(0), -1])
		code = F.selu(self.enc_linear_1(code))
		code = self.enc_linear_2(code)
		return code
	
	def decode(self, code):
		out = F.selu(self.dec_linear_1(code))
		out = F.sigmoid(self.dec_linear_2(out))
		out = out.view([code.size(0), NUM_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT])
		return out

