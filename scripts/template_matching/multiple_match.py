import cv2
import numpy as np
from matplotlib import pyplot as plt

# img_bgr = cv2.imread('coral_past_flip.jpg', 0)
# template = cv2.imread('intersection_pink_right.jpg', cv2.IMREAD_GRAYSCALE)

img_bgr = cv2.imread('coral_under2.jpg', cv2.IMREAD_UNCHANGED)
template = cv2.imread('pink_right_water.jpg', cv2.IMREAD_GRAYSCALE)

if img_bgr is None or template is None:
    exit("ERROR: failed to read images")

img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.75
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_bgr, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv2.imshow("pink_right1.png", template)
cv2.imshow('res.png',img_bgr)
cv2.waitKey(0)