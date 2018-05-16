import swag 
import cv2
import numpy as np

fname = '../../jackson-clips'

video = swag.VideoCapture(fname)

frameCount = 0

while True:
	ret, frame = video.read() 

	if ret == True: # ret = False indicates EOF
		cv2.putText(frame, "Frame " + str(frameCount), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255));
		cv2.imshow('frame', frame)
		frameCount += 1
		
		# hit 'p' key to pause and then 'p' again to play
		if cv2.waitKey(1) == ord('p'):
			while cv2.waitKey(1) != ord('p'): continue

		# hit 'q' key to close window
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	else:
		break

video.release()
cv2.destroyAllWindows()