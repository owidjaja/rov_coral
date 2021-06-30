import cv2
import numpy as np

ref = cv2.imread("../res/reference_coral_mask.jpg")
img = cv2.imread("../res/coral_mask.jpg")

kernel = np.ones((5,5),np.uint8)

ref_gradient = cv2.morphologyEx(ref, cv2.MORPH_GRADIENT, kernel)
img_gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

cv2.imshow("ref_g", ref_gradient)
cv2.imshow("img_g", img_gradient)

if cv2.waitKey(0) == ord('s'):
    cv2.imwrite("ref_coral_mask_gradient.jpg", ref_gradient)
    cv2.imwrite("coral_mask_gradient.jpg", img_gradient)

import cv2 as cv
gray = cv.cvtColor(ref,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
ref[dst>0.01*dst.max()]=[0,0,255]
cv.imshow('dst',ref)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()