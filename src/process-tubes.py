import pandas as pd
import cv2
from Tube import *

bb_fname = '../data/jackson-town-square-2017-12-14.csv'
bb = pd.read_csv(bb_fname, header = 0)

# find number of objects that have bounding boxes
obj_ids = bb['ind'].unique().flatten().tolist()

tubes = []

# for each object, produce a tube
for obj_id in obj_ids[0:2]:
	print('Creating tube for object', obj_id)
	tube = Tube(obj_id)
	tube.populate(bb)
	tubes.append(tube)
	print('Finished creating tube for object', obj_id, 'with', len(tube.rois), 'frames\n')

# try displaying ROIs
tube = tubes[1]
for frame in tube.rois:
	cv2.imshow('frame', frame)
	cv2.waitKey(0)

cv2.destroyAllWindows()