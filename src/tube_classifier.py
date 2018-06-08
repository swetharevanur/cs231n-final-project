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
import numpy as np
np.random.seed(695)
import csv

import VideoDataset
import ClassifierLoader
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
		return F.softmax(code)

def retrieve_metrics(loader, model):
	num_correct = 0
	num_samples = 0
	model.eval()  # set model to evaluation mode

	scores_arr = []#torch.Tensor((0, 2))
	y_arr = []

	with torch.no_grad():
		for _, (x, y) in enumerate(loader):
			x = x.unsqueeze_(1)
			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=torch.long)
			scores = model(x)
			scores_arr = scores_arr + list(scores[:, 0].cpu().numpy())
			y_arr += list(y.cpu().numpy())
			_, preds = scores.max(1)
			num_correct += (preds == y).sum()
			num_samples += preds.size(0)
	acc = float(num_correct) / num_samples
	# print('Val accuracy: %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

	# print(scores_arr.permute(0, 1)[:, 0].data)

	return scores_arr, y_arr, acc

# training model
def train_model(loader_train, loader_val, model, optimizer, results_fname, running_lowest_loss, lr, num_epochs = 10):
	loss_history = []
	
	for epoch in range(num_epochs):
		# print("\nEpoch %d" % epoch)
		for i, (x, y) in enumerate(loader_train): # ignore image labels
			x = x.unsqueeze_(1)
			x = x.to(device = device, dtype = dtype)
			y = y.to(device = device, dtype = dtype)
			scores = model(x)
			optimizer.zero_grad()
			loss = F.cross_entropy(scores, y.long())
			loss.backward()
			optimizer.step()
		# print("Epoch %d Train Loss = %.5f" % (epoch,loss.data[0]))
		with open(results_fname, 'a') as text_file:
			text_file.write("\nEpoch = %d" % epoch)
			text_file.write("\nEpoch Train Loss = %.5f" % loss.data[0])
			
		# save intermediate models
		if loss.data[0] < running_lowest_loss:
			running_lowest_loss = loss
			torch.save(model, 'models/epochs/classifier_' + str(lr) + '_epoch_optimized.pth')

		loss_history.append(float(loss.data[0]))
	# print("Final Loss = %.5f" % loss.data[0])	
	return loss_history


def create_model(loader_train, loader_val, loader_test, results_fname, epochs = 10, learning_rate = 0.001):
	print_every = 1
	model = CNNClassifier()
	model.to(device = device)
	optimizer = optim.Adam(model.parameters(), lr = learning_rate) # SGD or Adam?
	loss_history = train_model(loader_train, loader_val, model, optimizer, results_fname, np.inf, lr = learning_rate,  num_epochs = epochs)
	return loss_history, retrieve_metrics(loader_val, model), retrieve_metrics(loader_test, model)

def load(train_primitive_matrix, train_tubes, val_primitive_matrix, val_tubes, test_primitive_matrix, test_tubes):
	batch_size = 128

	train_data = ClassifierLoader.ClassifierLoader(train_primitive_matrix, train_tubes, NUM_FRAMES_PER_TUBE)
	loader_train = torch.utils.data.DataLoader(train_data, shuffle = True, \
		num_workers = 0, batch_size = batch_size)
	print(len(train_data), "examples in train set")


	val_data = ClassifierLoader.ClassifierLoader(val_primitive_matrix, val_tubes, NUM_FRAMES_PER_TUBE)
	loader_val = torch.utils.data.DataLoader(val_data, shuffle = True, \
		num_workers = 0, batch_size = batch_size)
	print(len(val_data), "examples in validation set")

	test_data = ClassifierLoader.ClassifierLoader(test_primitive_matrix, test_tubes, NUM_FRAMES_PER_TUBE)
	loader_test = torch.utils.data.DataLoader(test_data, shuffle = True, \
		num_workers = 0, batch_size = batch_size)
	print(len(test_data), "examples in test set")

	return loader_train, loader_val, loader_test


# hyperparameter sweeps with different learning rates
def tune(train_primitive_matrix, train_tubes, val_primitive_matrix, val_tubes, test_primitive_matrix, test_tubes, lr_arr, weak_volume):
	results_fname = 'models/results/classifier_epoch_exps.txt'

	metrics_fname = 'models/results/classifier_metrics.txt'
	# lr_arr = [1e-5, 1e-4, 1e-3, 1e-2]

	loader_train, loader_val, loader_test = load(train_primitive_matrix, train_tubes, val_primitive_matrix, val_tubes, test_primitive_matrix, test_tubes)

	lr_metrics = {}
	for lr in lr_arr:
		loss_history, (scores_arr, y_arr, val_acc), (_, _, test_acc) = create_model(loader_train, loader_val, loader_test, results_fname, epochs = 100, learning_rate = lr)
	
		with open(metrics_fname, 'a') as text_file:
			text_file.write("\nweak_volume = %d" % weak_volume)
			text_file.write("\nlr = %.5f\n" % lr)
			text_file.write(str(loss_history))
			text_file.write(str(scores_arr))
			text_file.write(str(y_arr))
			text_file.write(str(val_acc))
		                            
		print('Learning Rate = %.5f, loss = %.5f, valacc = %.3f' % (lr, loss_history[-1], val_acc))
		lr_metrics[lr] = loss_history, scores_arr, y_arr, val_acc, test_acc

	return lr_metrics 

# no need-we already have check_accuracy
# def test_model(x_test, y_test, model):
#	scores = model(x_test)
	# what do we test? Higher label?

