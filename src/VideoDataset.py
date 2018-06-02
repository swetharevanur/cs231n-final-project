import torch
from torch.utils.data.dataset import Dataset
import swag 
import cv2


class VideoDataset(Dataset):
	def __init__(self, fname, transform):
		self.fname = fname
		self.frame_per_clip = 150
		self.num_clips = 10 # 6490
		self.transform = transform

	def __getitem__(self, index):
		clip_number = index//self.frame_per_clip + 1 # indexed at 1
		clip_fname = self.fname + '/' + str(clip_number) + '.mp4'
		
		video = cv2.VideoCapture(clip_fname)
		video.set(1, index % self.frame_per_clip)
		ret, frame = video.read()
		

		if self.transform:
			#print(frame.shape)
			frame = cv2.resize(frame, dsize = (270, 480), interpolation = cv2.INTER_CUBIC)
			for tform in self.transform:
				frame = tform(frame)

		return (frame, 'train') # ignore labels!

	def __len__(self):
		return self.frame_per_clip * self.num_clips # number of frames in video

# frame_dataset = VideoDataset(fname = '../../jackson-clips', transform)

# for i in range(2197*150, len(frame_dataset)):
# 	sample, _ = frame_dataset[i]
# 	print(i)
# 	cv2.imshow('frame', sample)
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()
