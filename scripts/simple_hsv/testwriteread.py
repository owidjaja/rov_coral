import cv2
import numpy as np

img = np.zeros((300,300,3), dtype=np.uint8)
cv2.rectangle(img, (100,100), (200,200), (0,255,0), thickness=-1)

cv2.imwrite("testimg.jpg", img)

new = cv2.imread("testimg.jpg")
cv2.imshow("new", new)
cv2.waitKey(0)