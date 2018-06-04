import torch
from torch.utils.data.dataset import Dataset
import swag 
import cv2
import pandas as pd
import math

def resize_frame(roi):
	# pad ROI to the nearest multiple of 224
	new_dim = 224
	roi_height, roi_width, roi_channel = roi.shape
	vert_pad = (roi_height // new_dim + 1)*new_dim - roi_height
	horiz_pad = (roi_width // new_dim + 1)*new_dim - roi_width
	if roi_height % new_dim == 0: vert_pad = 0
	if roi_width % new_dim == 0: horiz_pad = 0

	top = math.ceil(vert_pad/2)
	bottom = vert_pad - top
	left = math.ceil(horiz_pad/2)
	right = horiz_pad - left

	padded_roi = cv2.copyMakeBorder(roi, top, bottom, left, right, 
	cv2.BORDER_CONSTANT, value = (0, 0, 0))

	H, W, C = padded_roi.shape
	
	# apply vectorized spatial max pooling								
	pool_kernel_H = H // new_dim
	pool_kernel_W = W // new_dim

	padded_roi = padded_roi.reshape(1, H // pool_kernel_H, pool_kernel_H, W // pool_kernel_W, pool_kernel_W, C)	
	padded_roi = padded_roi.max(axis = 2).max(axis = 3)
	padded_roi = padded_roi.squeeze(axis = 0)
	
	return padded_roi

class VideoDataset(Dataset):
	def __init__(self, fname, transform):
		self.fname = fname
		self.bb_fname = '../data/jackson-town-square-2017-12-14.csv'
		self.bb = pd.read_csv(self.bb_fname, header = 0)
		self.bb = self.bb.drop_duplicates(subset = 'frame') # for ConvAE training purposes, only keep first BB for frame
		self.frame_per_clip = 150
		self.num_clips = 30 # 6490
		self.transform = transform

	def extract_frame(self, index):
		clip_number = index//self.frame_per_clip + 1 # indexed at 1
		clip_fname = self.fname + '/' + str(clip_number) + '.mp4'
		
		video = cv2.VideoCapture(clip_fname)
		video.set(1, index % self.frame_per_clip)
		ret, frame = video.read()

		if ret == False: return None	
		return frame

	def __getitem__(self, index):
		frame = self.extract_frame(index)

		# crop the frame (based on first BB data)
		bb = self.bb
		bb_dims = ['xmin', 'ymin', 'xmax', 'ymax']

		x_min, y_min, x_max, y_max = [bb.iloc[index][dim].tolist() for dim in bb_dims]
		roi = frame[int(y_min):int(y_max), int(x_min):int(x_max), :] # crop

		padded_roi = resize_frame(roi)

		if self.transform:
			for tform in self.transform:
				padded_roi = tform(padded_roi)
		
		return (padded_roi, 'train') # ignore labels!

	def __len__(self):
		return self.frame_per_clip*self.num_clips
		# return self.bb['frame'].nunique()# number of frames in video with vehicles in them
		# self.frame_per_clip * self.num_clips # number of frames in video

# frame_dataset = VideoDataset(fname = '../../jackson-clips', transform)

# for i in range(2197*150, len(frame_dataset)):
# 	sample, _ = frame_dataset[i]
# 	print(i)
# 	cv2.imshow('frame', sample)
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()
