import cv2
import numpy as np

img = np.ones((50,50), np.uint8)

cv2.imwrite("out/testout.jpg", img)