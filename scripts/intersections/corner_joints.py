import cv2
import numpy as np

""" detect corner joints by intersection from hough lines 
    https://stackoverflow.com/questions/60633334/how-to-detect-corner-joints-that-connect-elements-on-images
"""

def auto_resize(img, target_width=800):
    # print("Original Dimension: ", img.shape)

    orig_height, orig_width = img.shape[:2]
    scale_ratio = target_width / orig_width

    new_width = int(img.shape[1] * (scale_ratio))
    new_height= int(img.shape[0] * (scale_ratio))
    dim = (new_width, new_height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # print("Resized Dimension: ", resized.shape, '\n')
    return resized

def cb_nothing(x):
    pass

# Load image, grayscale, Gaussian blur, Otsus threshold
image = cv2.imread('../res/test_target.jpg')
image = auto_resize(image)
# cv2.imshow("src", image)
# image = cv2.imread("../res/reference_coral_mask.jpg")
# cv2.imshow('image', image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)[1]
cv2.imshow('thresh', thresh)

cv2.namedWindow("Trackbar_Window", cv2.WINDOW_NORMAL)
cv2.createTrackbar("ksize1", "Trackbar_Window", 30, 50, cb_nothing)
cv2.createTrackbar("ksize2", "Trackbar_Window", 1, 10, cb_nothing)

while True:
    ksize1 = cv2.getTrackbarPos("ksize1", "Trackbar_Window")
    if ksize1<=0: ksize1 = 1
    ksize2 = cv2.getTrackbarPos("ksize2", "Trackbar_Window")
    if ksize2<=0: ksize2 = 1

    # Find horizonal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize1,ksize2))  # rectangular kernel
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cv2.imshow('horizontal', horizontal)

    # Find vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize2,ksize1))    # vertical rectangular kernel
    vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cv2.imshow('vertical', vertical)

    # Find joints
    joints = cv2.bitwise_and(horizontal, vertical)
    cv2.imshow('joints', joints)

    image2 = image.copy()       # to reset green circles

    # Find centroid of the joints
    cnts = cv2.findContours(joints, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]       # get contour return value, instead of ret

    for c in cnts:
        # Find centroid and draw center point
        M = cv2.moments(c)
        # print(M,"\n")
        if (M['m00']==0):
            # TODO: figure out what happens here
            # print("div by 0")
            continue
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.circle(image2, (cx, cy), 8, (36,255,12), -1)

    cv2.imshow('image2', image2)
    
    k = cv2.waitKey(1)
    if k == 27:    # esc to break
        break
    elif k == ord('s'):
        cv2.imwrite("mask_with_corners.jpg", image2)
        break