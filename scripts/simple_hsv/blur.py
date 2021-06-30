import cv2
import numpy as np

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

new = cv2.imread("../res/test_target.jpg")

kernel = np.ones((3,3), np.uint8)
blur = cv2.GaussianBlur(new, (51,51), 0)

cv2.imshow("blur", auto_resize(blur))

cv2.waitKey(0)