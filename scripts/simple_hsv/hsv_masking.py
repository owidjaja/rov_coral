import cv2

from scale_resizing import scale_resizing

img = cv2.imread("../res/coral_under1.jpg")

img_scaled = scale_resizing(img)

cv2.imshow("coral", img)
cv2.waitKey(0)