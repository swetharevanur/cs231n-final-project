# Adapted from: https://gist.github.com/okiriza/16ec1f29f5dd7b6d822a0a3f2af39274

import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import VideoDataset

# Using gpu
USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# device = torch.device('cpu')

print('using device:', device)

class AutoEncoder(nn.Module):
	def __init__(self, code_size):
		super().__init__()
		self.code_size = code_size
		
		# Encoder specification
		self.enc_cnn_1 = nn.Conv2d(NUM_CHANNELS, 10, kernel_size=5)
		self.enc_cnn_2 = nn.Conv2d(10, 20, kernel_size=5)
		self.enc_linear_1 = nn.Linear(117 * 64 * 20, self.code_size*2)
		self.enc_linear_2 = nn.Linear(self.code_size*2, self.code_size)
		
		# Decoder specification
		self.dec_linear_1 = nn.Linear(self.code_size, 160)
		self.dec_linear_2 = nn.Linear(160, IMAGE_SIZE)
		
	def forward(self, images):
		# image size is 1080 x 1920
		code = self.encode(images)
		out = self.decode(code)
		return out, code
	
	def encode(self, images):
		code = F.max_pool2d(images, 2)
		code = F.max_pool2d(code, 2)
		code = self.enc_cnn_1(code)
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
		out = out.view([code.size(0), NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH])
		return out
	

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
NUM_CHANNELS = 3
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS

# Hyperparameters
code_size = 100
num_epochs = 10
batch_size = 1
lr = 0.002
optimizer_cls = optim.Adam

# Load data
train_data = VideoDataset.VideoDataset(fname = '../../jackson-clips', transform=transforms.ToTensor())
print("About to train_loader")
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size,
num_workers = 0, drop_last=True)
# train_data = datasets.MNIST('~/data/mnist/', train=True , transform=transforms.ToTensor(), download = True)
# test_data  = datasets.MNIST('~/data/mnist/', train=False, transform=transforms.ToTensor())
# train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True)

# Instantiate model
autoencoder = AutoEncoder(code_size)
loss_fn = nn.BCELoss()
optimizer = optimizer_cls(autoencoder.parameters(), lr=lr)
torch.save(autoencoder, 'models/autoencoder.pt')

autoencoder = autoencoder.cuda()
# Training loop
for epoch in range(num_epochs):
	print("Epoch %d" % epoch)

	for i, (images, _) in enumerate(train_loader): # Ignore image labels
		print(i)
		images = images.to(device = device, dtype = dtype)
		out, code = autoencoder(images)
		#out = out.to(device = device, dtype = dtype)
		#code = code.to(device = device, dtype = dtype)
		optimizer.zero_grad()
		loss = loss_fn(out, images)
		loss.backward()
		optimizer.step()
		print("Loop Loss = %.3f" % loss.data[0])
		
	print("Loss = %.3f" % loss.data[0])

# Try reconstructing on test data
# test_image = random.choice(train_data)
# test_image = Variable(test_image.view([1, 1, IMAGE_WIDTH, IMAGE_HEIGHT]))
# test_reconst, _ = autoencoder(test_image)

# torchvision.utils.save_image(test_image.data, 'orig.png')
# torchvision.utils.save_image(test_reconst.data, 'reconst.png')

torch.save(autoencoder, 'models/autoencoder.pt')

