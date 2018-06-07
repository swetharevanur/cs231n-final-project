#import torch
#torch.multiprocessing.get_context("spawn")
import torch
# torch.multiprocessing.set_start_method("spawn")
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import VideoDataset
import ClassifierLoader
import numpy as np
from process_tubes import NUM_FRAMES_PER_TUBE

USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Using device:', device)

# input: 30x512 matrix for every action tube
class CNNClassifier(nn.Module):
	def __init__(self):
		super().__init__()
		# network specification
		self.cnn_1 = nn.Conv2d(1, 20, kernel_size=3, padding = 1)
		nn.init.kaiming_normal_(self.cnn_1.weight)
		self.batch_1 = nn.BatchNorm2d(20)
		self.cnn_2 = nn.Conv2d(20, 20, kernel_size=3, padding = 1)
		nn.init.kaiming_normal_(self.cnn_2.weight)
		self.batch_2 = nn.BatchNorm2d(20)
		self.cnn_3 = nn.Conv2d(20, 20, kernel_size=3, padding = 1)
		nn.init.kaiming_normal_(self.cnn_3.weight)
		self.batch_3 = nn.BatchNorm2d(20)
		self.linear_1 = nn.Linear(20 * 3 * 64, 100)
		nn.init.kaiming_normal_(self.linear_1.weight)
		self.linear_2 = nn.Linear(100, 2)
		nn.init.kaiming_normal_(self.linear_2.weight)
			
	def forward(self, tube):
		code = self.cnn_1(tube)
		code = F.relu(F.max_pool2d(code, 2))
		code = self.batch_1(code)
		code = self.cnn_2(code)
		code = F.relu(F.max_pool2d(code, 2))
		code = self.batch_2(code)
		code = self.cnn_3(code)
		code = F.relu(F.max_pool2d(code, 2))
		code = self.batch_3(code)
		code = code.view([tube.size(0), -1])
		code = F.relu(self.linear_1(code))
		code = self.linear_2(code)
		return F.sigmoid(code)

def check_accuracy(loader_val, model):
	num_correct = 0
	num_samples = 0
	model.eval()  # set model to evaluation mode
	with torch.no_grad():
		# insert loader line here
		for _, (x, y) in enumerate(loader_val):
			x = x.unsqueeze_(1)
			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=torch.long)
			scores = model(x)
			_, preds = scores.max(1)
			num_correct += (preds == y).sum()
			num_samples += preds.size(0)
	acc = float(num_correct) / num_samples
	print('Classified %d / %d correctly (%.2f)' % (num_correct, num_samples, 100 * acc))

# training model
def train_model(loader_train, loader_val, model, optimizer, results_fname, running_lowest_loss, lr, num_epochs = 10):
	for epoch in range(num_epochs):
		print("Epoch %d" % epoch)
		for i, (x, y) in enumerate(loader_train): # ignore image labels
			x = x.unsqueeze_(1)
			x = x.to(device = device, dtype = dtype)
			y = y.to(device = device, dtype = dtype)
			scores = model(x)
			optimizer.zero_grad()
			loss = F.cross_entropy(scores, y.long())
			loss.backward()
			optimizer.step()
		print("Epoch Loss = %.5f" % loss.data[0])
		with open(results_fname, 'a') as text_file:
			text_file.write("\nEpoch = %d" % epoch)
			text_file.write("\nEpoch Loss = %.5f" % loss.data[0])
		check_accuracy(loader_val, model)


		# save intermediate models
		if loss.data[0] < running_lowest_loss:
			running_lowest_loss = loss
			torch.save(model, 'models/epochs/classifier_' + str(lr) + '_epoch_optimized.pth')

	print("Final Loss = %.5f" % loss.data[0])	
	return loss.data[0]


'''
def train_model(loader_train, loader_val, model, optimizer, lr, epochs = 10):
	# model = model.to(device=device)  # move the model parameters to CPU/GPU
	running_lowest_loss = np.inf
	for e in range(epochs):
		# insert loader code
		for _, (x, y) in enumerate(loader_train):
			model.train()  # put model to training mode
			x = x.unsqueeze_(1)
			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=torch.long)
			scores = model(x)
			loss = F.cross_entropy(scores, y) # this works fine, also could use score for wrong class
			# zero out all of the gradients for the variables which the optimizer
			# will update.
			optimizer.zero_grad()
			# this is the backwards pass: compute the gradient of the loss with
			# respect to each  parameter of the model.
			loss.backward()
			# actually update the parameters of the model using the gradients
			# computed by the backwards pass.
			optimizer.step()
		print('Epoch %d, loss = %.4f' % (e, loss.item()))
		with open(results_fname, 'a') as text_file:
			text_file.write("\nEpoch = %d" % epoch)
			text_file.write("\nEpoch Loss = %.5f" % loss.data[0])
		check_accuracy(x_val, y_val, model)
		print()

		if loss.data[0] < running_lowest_loss:
			running_lowest_loss = loss
			torch.save(model, 'models/epochs/classifier_' + lr + '_epoch_optimized.pth')

	del model
	return(loss.item())
'''

def create_model(loader_train, loader_val, results_fname, epochs = 10, learning_rate = 0.001):
	print_every = 1
	model = CNNClassifier()
	model.to(device = device)
	optimizer = optim.Adam(model.parameters(), lr = learning_rate) # SGD or Adam?
	loss = train_model(loader_train, loader_val, model, optimizer, results_fname, np.inf, lr = learning_rate,  num_epochs = epochs)
	return loss

def load(train_primitive_matrix, train_tubes, val_primitive_matrix, val_tubes):
	batch_size = 128

	train_data = ClassifierLoader.ClassifierLoader(train_primitive_matrix, train_tubes, 'train', NUM_FRAMES_PER_TUBE)
	loader_train = torch.utils.data.DataLoader(train_data, shuffle = True, \
		num_workers = 0, batch_size = batch_size)

	val_data = ClassifierLoader.ClassifierLoader(val_primitive_matrix, val_tubes, 'val', NUM_FRAMES_PER_TUBE)
	loader_val = torch.utils.data.DataLoader(val_data, shuffle = True, \
		num_workers = 0, batch_size = batch_size)

	return loader_train, loader_val


# hyperparameter sweeps with different learning rates
def tune(train_primitive_matrix, train_tubes, val_primitive_matrix, val_tubes):
	results_fname = 'models/results/classifier_epoch_exps.txt'
	lr_arr = [1e-2, 1e-3, 1e-4, 1e-5]

	loader_train, loader_val = load(train_primitive_matrix, train_tubes, val_primitive_matrix, val_tubes)
	
	for lr in lr_arr:
		loss = create_model(loader_train, loader_val, results_fname, epochs = 10, learning_rate = lr)
		print('Learning Rate = %.5f, loss = %.4f' % (lr, loss))

# no need-we already have check_accuracy
# def test_model(x_test, y_test, model):
#	scores = model(x_test)
	# what do we test? Higher label?

