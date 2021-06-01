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
