# Reads in stream of 5s clips from jackson-clips directory
# Used to create visualizations for paper and poster

import swag 
import cv2
import pandas as pd

fname = '../../jackson-clips'
video = swag.VideoCapture(fname)
bb_fname = '../data/jackson-town-square-2017-12-14.csv'
bb = pd.read_csv(bb_fname, header = 0)

# find number of objects that have bounding boxes
# obj_ids = bb['ind'].unique().flatten().tolist()

frame_num = 467
frames_of_interest = bb.loc[bb['frame'] == frame_num]
bb_dims = ['xmin', 'ymin', 'xmax', 'ymax']
for _, frame_row in frames_of_interest.iterrows():
	x_min, y_min, x_max, y_max = [frame_row[dim] for dim in bb_dims]

	# read in frame of interest
	fname = '../../jackson-clips'
	video = swag.VideoCapture(fname)
	video.set(1, frame_row['frame'])
	ret, frame = video.read()
	if ret == False: break # EOF reached

	roi = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
	cv2.imshow('frame', roi)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	# roi = cv2.resize(roi, (960, 540))

# while True:
# 	ret, frame = video.read() 

# 	if ret == True: # ret = False indicates EOF
# 		cv2.putText(frame, "Frame " + str(frameCount), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255));
# 		frame = cv2.resize(frame, (960, 540))

# 		cv2.imshow('frame', frame)
# 		frameCount += 1
		
# 		# hit 'p' key to pause and then 'p' again to play
# 		if cv2.waitKey(1) == ord('p'):
# 			while cv2.waitKey(1) != ord('p'): continue

# 		# hit 'q' key to close window
# 		if cv2.waitKey(1) & 0xFF == ord('q'):
# 			break

# 	else:
# 		break

# video.release()
# cv2.destroyAllWindows()