import swag 
import cv2
import numpy as np

fname = '../../jackson-clips'

video = swag.VideoCapture(fname)
 
while (True):
	ret, frame = video.read() 

	if ret == True: # ret = False indicates EOF
		cv2.imshow('frame', frame)
		# hit 'q' key to close window
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break

video.release()
cv2.destroyAllWindows()