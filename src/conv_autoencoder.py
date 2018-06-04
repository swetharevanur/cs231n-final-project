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
from ConvAE import AutoEncoder

# Using gpu
USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# device = torch.device('cpu')

print('using device:', device)	

IMAGE_WIDTH = 100 # 1920
IMAGE_HEIGHT = 100 # 1080
NUM_CHANNELS = 3
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS

# Hyperparameters
code_size = 50
num_epochs = 3
batch_size = 128
lr = 0.002
optimizer_cls = optim.Adam

# Load data
#transforms.Compose([
#transforms.Resize(size=(270, 480)),
#transforms.ToTensor()
#])
train_data = VideoDataset.VideoDataset(fname = '../../jackson-clips',transform=[transforms.ToTensor()])
print("About to train_loader")
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers = 8, drop_last=True)
# train_data = datasets.MNIST('~/data/mnist/', train=True , transform=transforms.ToTensor(), download = True)
# test_data  = datasets.MNIST('~/data/mnist/', train=False, transform=transforms.ToTensor())
# train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True)

# Instantiate model
autoencoder = AutoEncoder(code_size)
loss_fn = nn.BCELoss()
optimizer = optimizer_cls(autoencoder.parameters(), lr=lr)

autoencoder = autoencoder.cuda()
# Training loop
for epoch in range(num_epochs):
	print("Epoch %d" % epoch)

	for i, (images, _) in enumerate(train_loader): # Ignore image labels
		print(i)
		images = images.to(device = device, dtype = dtype)
		out, code = autoencoder(images)
		optimizer.zero_grad()
		loss = loss_fn(out, images)
		loss.backward()
		optimizer.step()
		print("Loop Loss = %.3f" % loss.data[0])
		
	print("Loss = %.3f" % loss.data[0])

torch.save(autoencoder, 'models/autoencoder.pth')

