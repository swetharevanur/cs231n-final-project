import pandas as pd
import cv2

from loader_utils import *
from Tube import *

def get_objects(num_objects = (30, 300, 70)):
	bb = preprocess_bb()
	obj_ids = bb['ind'].unique().flatten().tolist()
	# randomly sample
	obj_subset = np.random.choice(obj_ids, num_objects, replace = False)	
	return obj_subset

def tube_loader(mode = 'res18', obj_subset):	
	bb = preprocess_bb()
	obj_subset = get_objects(bb, num_objects)

	# encode frames
	encoder = init_encoder(mode)
	codes = []
	for obj_id in obj_subset: 
		print("\nCreating tubes for object index", obj_id)
		tube = Tube(obj_id)
		frames_of_interest = tube.get_frames(bb)
		count = 0
		for i, frame_row in frames_of_interest.iterrows():
			ret, frame = tube.read_frame(frame_row)
			if ret == False: break # EOF reached

			frame = preprocess_frame(bb, frame_row['frame'], frame)
			frameTensor = frame2tensor(frame)
			print(frameTensor.shape)
			codes.append(encode(encoder, frameTensor, mode))
			count += 1
			print("Encoded frame", i)
			if count == 5: break
			
	train_primitive_matrix = torch.stack(codes, 0).squeeze(1)

def produce_tubes(weak_labels_fname, obj_subset):
	bb = preprocess_bb()
	    
	tubes = []
	# for each object, produce a tube
	for obj_id in obj_subset: 
		print('Creating tube for object', obj_id)
		tube = Tube(obj_id)
	
		# add RoIs to tube and find tube labels
		tube.populate(bb, weak_labels_fname)
		# perform spatial max pooling
		tube.spatial_pooling()
		# perform temporal max pooling
		tube.temporal_pooling()

		tubes.append(tube)
		print('Finished creating tube for object', obj_id, 'with', len(tube.rois), 'frames\n')

		# while True:
		#	cv2.imshow('frame', tubes[0].summary_frame)
		#	if cv2.waitKey(1) == ord('q'): break
		# cv2.destroyAllWindows()

		# displaying ROIs
		# tubes[0].display()

obj_subset = get_objects()
tube_loader(obj_subset) # pass train primitive matrix to Reef
