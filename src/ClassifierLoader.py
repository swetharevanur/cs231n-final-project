import torch
from torch.utils.data.dataset import Dataset
import swag
import cv2


class ClassifierLoader(Dataset):
	def __init__(self, primitive_matrix, tubes, num_frames_per_tube):
		self.primitive_matrix = primitive_matrix
		#.to(device = torch.device('cuda'), dtype = torch.float32)
		self.tubes = tubes
		self.num_frames_per_tube = num_frames_per_tube

	def __getitem__(self, index):
		start_ind = self.num_frames_per_tube * index
		end_ind = self.num_frames_per_tube * (index + 1)
		tube_matrix = self.primitive_matrix[start_ind:end_ind, :]
		
		label = self.tubes[index].pred_vehicle
		if label is None:
			label = self.tubes[index].true_vehicle
		
		return (tube_matrix, label)

	def __len__(self):
		return len(self.tubes) 


