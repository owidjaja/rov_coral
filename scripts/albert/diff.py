import cv2
import numpy as np

old_pink = cv2.imread("old_pink_mask.jpg")
old_white = cv2.imread("old_white_mask.jpg")
new_pink = cv2.imread("new_pink_mask.jpg")
new_white = cv2.imread("new_white_mask.jpg")

grey_levels = 256
# Generate a test image
ret, test_image = cv2.threshold(old_pink, 32, 255, cv2.THRESH_BINARY)

# Define the window size
windowsize_r = 50
windowsize_c = 50

# Crop out the window and calculate the histogram
for r in range(0,test_image.shape[0] - windowsize_r, windowsize_r):
    for c in range(0,test_image.shape[1] - windowsize_c, windowsize_c):
        window = test_image[r:r+windowsize_r,c:c+windowsize_c]
        hist = np.histogram(window,bins=grey_levels)