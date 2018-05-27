import swag 
import cv2

import numpy as np
import scipy
import csv
import sklearn.cross_validation
import pandas as pd

from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from get_frames import getVehicleFrames


def parse_file(filename):
	frameLabels = []
	with open(filename, 'rt') as f:
		framereader = csv.reader(f, delimiter = ',')
		next(framereader)
		for frame in framereader:
			frameLabels.append((frame[0], frame[1]))
	return frameLabels

def split_data(X, y):
	np.random.seed(1234)
	num_sample = np.shape(X)[0]
	num_test = num_sample // 5
	
	flattened_list = [y for x in list_of_lists for y in x]


	X_test = X[0:num_test]
	X_train = X[num_test:]

	y_test = y[0:num_test]
	y_train = y[num_test:]

	# split dev/test
	test_ratio = 0.2
	X_tr, X_te, y_tr, y_te = \
		sklearn.cross_validation.train_test_split(X_train, y_train, test_size = test_ratio)

	return np.array(X_tr.todense()), np.array(X_te.todense()), np.array(X_test.todense()), \
		np.array(y_tr), np.array(y_te), np.array(y_test)


class DataLoader(object):
	""" A class to load in appropriate numpy arrays
	"""

	def __init__(self):
		self.rois = [] # list of regions of interest
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

		# crop the frames
		frameNumsToLoad = getVehicleFrames(data_path + fname1)[0][0:10]

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

			roi = frame[int(y_min):int(y_max), int(x_min):int(x_max)] # crop
			self.rois.append(roi)

			if frameNum%10 == 0:
				print(frameNum)

		# get labels associated with each frame
		labels = parse_file(data_path + fname1)

		# featurize plots  
		# vectorizer = CountVectorizer(min_df=1, binary=True, \
		# 	decode_error='ignore', strip_accents='ascii', ngram_range=(1,2))
		# X = vectorizer.fit_transform(plots)
		# valid_feats = np.where(np.sum(X,0)> 2)[1]
		# X = X[:,valid_feats]

		# split dataset into train, val, test
		train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
			train_ground, val_ground, test_ground = split_data(self.rois, labels)

		#Prune Feature Space
		common_idx = self.prune_features(val_primitive_matrix, train_primitive_matrix)
		return train_primitive_matrix[:,common_idx], val_primitive_matrix[:,common_idx], test_primitive_matrix[:,common_idx], \
			np.array(train_ground), np.array(val_ground), np.array(test_ground)


