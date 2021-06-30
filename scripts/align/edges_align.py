import cv2
import numpy as np

ref = cv2.imread("../res/reference_coral_mask.jpg")
img = cv2.imread("../res/coral_mask.jpg")

ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("ref_gray", ref_gray)
# cv2.imshow("img_gray", img_gray)

ref_gray = cv2.morphologyEx(ref_gray, cv2.MORPH_CLOSE, np.ones((5,5),dtype=np.uint8))

ref_cnts, ref_hierarchy = cv2.findContours(ref_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
img_cnts, img_hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in ref_cnts:
    epsilon = 0.0008 * cv2.arcLength(cnt, True)
    approximations = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(ref, [approximations], 0, (255,0,0), 3)

for cnt in img_cnts:
    epsilon = 0.001 * cv2.arcLength(cnt, True)
    approximations = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(img, [approximations], 0, (255,0,0), 2)

cv2.imshow("ref", ref)
cv2.imshow("img", img)

cv2.waitKey(0)