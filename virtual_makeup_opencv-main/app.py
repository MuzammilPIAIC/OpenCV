import cv2
import dlib
import numpy as np

def nothing(x):
	pass
# set up the 68 point facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cv2.namedWindow('controls')
    #create trackbar in 'controls' window with name 'r''
cv2.createTrackbar('r','controls',15,255,nothing)
cv2.createTrackbar('b','controls',15,255,nothing)
cv2.createTrackbar("hue", 'controls', 1, 180, nothing)
cv2.createTrackbar("sat", 'controls', 65, 155, nothing)

