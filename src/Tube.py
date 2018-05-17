# Creates normalized actions tubes for objects from bounding box data
import swag 
import cv2
from operator import itemgetter
import math

class Tube():
	def __init__(self, obj_id):
		self.id = obj_id
		self.rois = [] # list of regions of interest
		self.pooled_rois = [] # RoIs after spatial max pooling across all frames

	def populate(self, bb):
		frames_of_interest = bb.loc[bb['ind'] == self.id]

		# crop frames
		bb_dims = ['xmin', 'ymin', 'xmax', 'ymax']
		for index, frame in frames_of_interest.iterrows():
			x_min, y_min, x_max, y_max = [frame[dim] for dim in bb_dims]

			# read in frame of interest
			fname = '../../jackson-clips'
			video = swag.VideoCapture(fname)
			video.set(1, frame['frame'])
			ret, frame = video.read()
			if ret == False: break # EOF reached

			roi = frame[int(y_min):int(y_max), int(x_min):int(x_max)] # crop
			self.rois.append(roi)

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
			print(frame_height, frame_width)

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
			same_size = pool_kernel_dim == stride
			tiles = H % pool_kernel_dim == 0 and W % pool_kernel_dim == 0
			if same_size and tiles:
				padded_frame = padded_frame.reshape(1, C, H // pool_kernel_dim, pool_kernel_dim,
						   W // pool_kernel_dim, pool_kernel_dim)
				padded_frame = padded_frame.max(axis = 3).max(axis = 4)
			else:
				assert "Incorrect kernel size or stride for spatial max pooling"

			print(padded_frame.shape[0], padded_frame.shape[1])
			self.pooled_rois.append(padded_frame)


	def temporal_pooling(self):
		return




