import swag 
import cv2

def readSpecificFrame(frameNum, frameCache):
	fname = '../../jackson-clips'
	video = swag.VideoCapture(fname)

	video.set(2, frameNum)
	ret, frame = video.read()

	frameCache.append(frame)