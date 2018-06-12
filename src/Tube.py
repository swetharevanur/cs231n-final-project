# Creates normalized actions tubes for objects from bounding box data
import swag 
import cv2
from operator import itemgetter
import math
import numpy as np

class Tube():
	def __init__(self, obj_id):
		self.id = obj_id
		self.rois = [] # list of regions of interest
		self.pooled_rois = [] # RoIs after spatial max pooling across all frames
		self.summary_frame = None
		self.numFrames = 0
		self.tube_pred_labels = []
		self.tube_true_labels = []
		self.pred_vehicle = None
		self.true_vehicle = None


	def populate(self, bb, weak_labels_fname):
		frames_of_interest = bb.loc[bb['ind'] == self.id]

		# load in Reef weak labels
		weak_labels = np.load(weak_labels_fname)

		# crop frames
		bb_dims = ['xmin', 'ymin', 'xmax', 'ymax']
		for _, frame_row in frames_of_interest.iterrows():
			x_min, y_min, x_max, y_max = [frame_row[dim] for dim in bb_dims]

			# read in frame of interest
			fname = '../../jackson-clips'
			video = swag.VideoCapture(fname)
			video.set(1, frame_row['frame'])
			ret, frame = video.read()
			if ret == False: break # EOF reached

			# get weak label for frame
			frameNum = frame_row['frame']
			print(frameNum)
			print(weak_labels)
			self.tube_pred_labels.append(weak_labels[frameNum])

			# get true label for frame
			self.tube_true_labels.append(-1 if frame_row['object_name'] == 'truck' else 1)

			# crop
			roi = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
			self.rois.append(roi)

			self.numFrames += 1

		video.release()

	def display(self):
		for frame in self.rois:
			cv2.imshow('frame', frame)
			cv2.waitKey(0)
		cv2.destroyAllWindows()

	def spatial_pooling(self):
		new_dim = 20
		pool_kernel_dim = 4
		stride = 4

		for frame in self.rois:
			frame_height, frame_width, frame_channel = frame.shape

			# zero pad each frame so that they all have height/width that is a multiple of new_dim
			# multiple must be larger than frame dims
			vert_pad = (frame_height // new_dim + 1)*new_dim - frame_height
			horiz_pad = (frame_width // new_dim + 1)*new_dim - frame_width
			if frame_height % new_dim == 0: vert_pad = 0
			if frame_width % new_dim == 0: horiz_pad = 0

			top = math.ceil(vert_pad/2)
			bottom = vert_pad - top
			left = math.ceil(horiz_pad/2)
			right = horiz_pad - left

			padded_frame = cv2.copyMakeBorder(frame, top, bottom, left, right, 
				cv2.BORDER_CONSTANT, value = (0, 0, 0))

			H, W, C = padded_frame.shape

			# apply vectorized spatial max pooling
			pool_kernel_H = H // new_dim
			pool_kernel_W = W // new_dim

			padded_frame = padded_frame.reshape(1, H // pool_kernel_H, pool_kernel_H,
					   W // pool_kernel_W, pool_kernel_W, C)
			padded_frame = padded_frame.max(axis = 2).max(axis = 3)
			padded_frame = padded_frame.squeeze(axis = 0)

			self.pooled_rois.append(padded_frame)

	def temporal_pooling(self):
		# stacks spatially max pooled frames
		temp_pooled_frame = np.stack(self.pooled_rois, axis = 0)
		t, x, y, c = temp_pooled_frame.shape

		temp_pooled_frame = temp_pooled_frame.max(axis = 0)
		self.summary_frame = temp_pooled_frame

	def label(self):
		# assign majority label to tube
		print(self.tube_pred_labels)
		self.pred_vehicle = -1 if sum(self.tube_pred_labels)/self.numFrames < 0.5 else 1
		self.true_vehicle = max(set(self.tube_true_labels), key = self.tube_true_labels.count)

