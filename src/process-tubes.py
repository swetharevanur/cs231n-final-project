import pandas as pd
import numpy as np
import cv2
from Tube import *

weak_labels_fname = '../data/weak_labels_res18_reef.npy'

bb_fname = '../data/jackson-town-square-2017-12-14.csv'
bb = pd.read_csv(bb_fname, header = 0)

# find number of objects that have bounding boxes
obj_ids = bb['ind'].unique().flatten().tolist()

# randomly sample objects
num_objects = 10
obj_subset = np.random.choice(obj_ids, num_objects, replace = False)

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

	tube.label()

	tubes.append(tube)
	print('Finished creating tube for object', obj_id, 'with', len(tube.rois), 'frames')
	print("Pred: ", tube.pred_vehicle)
	print("True: ", tube.true_vehicle)

# while True:
# 	cv2.imshow('frame', tubes[0].summary_frame)
# 	if cv2.waitKey(1) == ord('q'): break
# cv2.destroyAllWindows()

# displaying ROIs
# tubes[0].display()
