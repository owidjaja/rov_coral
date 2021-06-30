import cv2
import numpy as np

""" study cv2.getStructuringElement
    understanding: just allows us to produce kernels with irregular shapes, instead of just squares with np.ones 
"""

def cb_nothing(x):
    pass

cv2.namedWindow("t_win", cv2.WINDOW_NORMAL)
cv2.createTrackbar("ksize1", "t_win", 5, 30, cb_nothing)
cv2.createTrackbar("ksize2", "t_win", 5, 30, cb_nothing)

while True:
    ksize1 = cv2.getTrackbarPos("ksize1", "t_win")
    ksize2 = cv2.getTrackbarPos("ksize2", "t_win")

    print("\nrect\n", cv2.getStructuringElement(cv2.MORPH_RECT,(ksize1,ksize2)))

    print("\ncross\n", cv2.getStructuringElement(cv2.MORPH_CROSS,(ksize1,ksize2)))
    
    print("\nellipse\n", cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize1,ksize2)))


    k = cv2.waitKey(0)
    if k == 27:
        continue
    elif k == ord('q'):
        break