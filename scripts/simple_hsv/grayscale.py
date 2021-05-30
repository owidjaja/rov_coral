import cv2
import numpy as np

def cb_nothing(x):
    pass

# main
src = cv2.imread("../res/coral_under3.jpg")
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray", gray)

cv2.namedWindow("Trackbar_Window")
cv2.createTrackbar("gray", "Trackbar_Window", 30, 255, cb_nothing)

while(True):
    thresh_val = cv2.getTrackbarPos("gray", "Trackbar_Window")
    ret, out = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    cv2.imshow("OUTPUT", out)

    erode = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((5,5), np.uint8) )
    cv2.imshow("ERODE", erode)

    if cv2.waitKey(1) == 'q':
        exit()