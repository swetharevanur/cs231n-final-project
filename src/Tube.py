import swag 
import cv2

class Tube():
	def __init__(self, obj_id):
		self.id = obj_id
		self.rois = [] # list of regions of interest

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

			roi = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
			self.rois.append(roi)

		video.release()