import cv2
import numpy as np
from matplotlib import pyplot as plt

# img = cv22.imread('coral_past_flip.jpg',0)
# template = cv2.imread('intersection_pink_right.jpg',0)

img = cv2.imread('coral_under2.jpg', 0)
template = cv2.imread('pink-removebg-preview_1_40x51.png', 0)
if img is None or template is None:
    exit("ERROR: failed to read images")

img2 = img.copy()
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)
    
    plt.subplot(131), plt.imshow(template, cmap='gray')
    plt.title('Template'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()