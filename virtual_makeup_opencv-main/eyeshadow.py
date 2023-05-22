import cv2
import numpy as np

img = cv2.imread('f.jpg')
# img = img.resize()
shape = img.shape
img2 = np.zeros(shape, np.uint8)
circle = cv2.circle(img2,(310,600), 40, (0,0,255), 2)
points = np.array([[910, 641], [206, 632], [696, 488]])
poly =  cv2.fillPoly(img2, [points], (255, 255, 255))
blur = cv2.blur(circle,(70, 70))
#dst = cv2.addWeighted(blur, 1, img, 1, 0.0)


# cv2.imshow('img', img)
# cv2.imshow('img2', img2)
# cv2.imshow('img3', blur)
cv2.imshow('img4', poly)
# cv2.imshow('img4', dst)
cv2.waitKey(0)
