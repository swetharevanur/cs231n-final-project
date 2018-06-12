import pandas as pd
import cv2

from loader_utils import *
from Tube import *
import VideoDataset

NUM_FRAMES_PER_TUBE = 30
NUM_CLASSES = 2

def get_objects(num_objects = (30, 300, 70)):
	bb = preprocess_bb(trunc_margin = NUM_FRAMES_PER_TUBE)
	obj_ids = bb['ind'].unique().flatten().tolist()

	total = sum(num_objects)
	num_labeled, num_unlabeled, num_test = num_objects	

	# determine class counts
	vehicle_frames = {'car': [], 'truck': []}
	for obj in obj_ids:
		vehicle_type = bb.loc[bb['ind'] == obj]['object_name'].to_string()
		vehicle_type = vehicle_type.split(' ')[-1]
		vehicle_frames[vehicle_type].append(obj)
	car_count = len(vehicle_frames['car'])
	truck_count = len(vehicle_frames['truck'])

	# force class balancing by randomly popping from the larger class
	if car_count > truck_count: # pop from car
		diff = car_count - truck_count
		if truck_count < total // NUM_CLASSES: # there are more car than trucks in bb
			diff = car_count - total + truck_count
		obj_remove = np.random.choice(vehicle_frames['car'], diff, replace = False)
	elif car_count < truck_count: # pop from truck
		diff = truck_count - car_count
		obj_remove = np.random.choice(vehicle_frames['truck'], diff, replace = False)
	obj_ids = list(set(obj_ids) - set(obj_remove))

	# randomly sample objects
	obj_subset = np.random.choice(obj_ids, sum(num_objects), replace = False)	
	
	return obj_subset[0:num_labeled], obj_subset[num_labeled: num_labeled + num_unlabeled], obj_subset[num_labeled + num_unlabeled:]


def tube_loader(obj_subset, mode = 'res18', label = False):	
	bb = preprocess_bb(trunc_margin = NUM_FRAMES_PER_TUBE)
	print("Producing action tubes...")	
	# encode frames
	encoder = init_encoder(mode)
	codes = []
	tubes = []
	for obj_id in obj_subset: 
		tube = Tube(obj_id)
		frames_of_interest = tube.get_frames(bb)

		# randomly sample frames
		if len(frames_of_interest) > NUM_FRAMES_PER_TUBE:
			frames_of_interest = frames_of_interest.sample(NUM_FRAMES_PER_TUBE)
		
		print("Creating action tube for object index", obj_id)
		# each tube is 30 x 512
		for _, frame_row in frames_of_interest.iterrows():
			ret, frame = tube.read_frame(frame_row)
			if ret == False: break # EOF reached
			frame_num = frame_row['frame']
			tube.sampled_frames.append(frame_num)
			frame = preprocess_frame(bb, frame_num, frame)
			frameTensor = frame2tensor(frame)
			codes.append(encode(encoder, frameTensor, mode))
			if label:
				tube.tube_true_labels.append(-1 if frame_row['object_name'] == 'truck' else 1)
		if label:
			tube.true_vehicle = max(set(tube.tube_true_labels), key = tube.tube_true_labels.count)
		tubes.append(tube)
	
	primitive_matrix = torch.stack(codes, 0).squeeze(1)
	return primitive_matrix, tubes

'''
def produce_tubes(weak_labels_fname, obj_subset):
	bb = preprocess_bb()
	tubes = []
	# for each object, produce a tube
	for obj_id in obj_subset: 
		print('Creating tube for object index', obj_id)
		tube = Tube(obj_id)
	
		# add RoIs to tube and find tube labels
		tube.populate(bb, weak_labels_fname)
		# perform spatial max pooling
		tube.spatial_pooling()
		# perform temporal max pooling
		tube.temporal_pooling()

		tubes.append(tube)
		print('Finished creating tube for object', obj_id, 'with', len(tube.rois), 'frames\n')	
'''
