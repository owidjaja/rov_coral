#!/usr/bin/env python
import cv2

def pad_images_to_same_size(images):
    """
    :param images: sequence of images
    :return: list of images padded so that all images have same width and height (max width and height are used)
    """
    width_max = 0
    height_max = 0
    for img in images:
        h, w = img.shape[:2]
        width_max = max(width_max, w)
        height_max = max(height_max, h)

    images_padded = []
    for img in images:
        h, w = img.shape[:2]
        diff_vert = height_max - h
        pad_top = diff_vert//2
        pad_bottom = diff_vert - pad_top
        diff_hori = width_max - w
        pad_left = diff_hori//2
        pad_right = diff_hori - pad_left
        img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        assert img_padded.shape[:2] == (height_max, width_max)
        images_padded.append(img_padded)

    return images_padded


src_arr = ['coral-colony-test-1_51268948073_o.jpg','coral-colony-test-2_51268762126_o.jpg','coral-colony-test-3_51269790240_o.jpg','Coral Colony F.png']
RATIO = 0.50

# old = cv2.imread("./sample/" + src_arr[1])
# old = cv2.resize(old, ( int(old.shape[1]*RATIO), int(old.shape[0]*RATIO) ), interpolation=cv2.INTER_AREA)
# # cv2.imshow("old", old)
# print("old.shape:", old.shape)

# new = cv2.imread("./sample/" + src_arr[2])
# new = cv2.resize(new, ( int(new.shape[1]*RATIO), int(new.shape[0]*RATIO) ), interpolation=cv2.INTER_AREA)
# # cv2.imshow("new", new)
# print("new.shape:", new.shape)

mat_arr = []
for path in src_arr:
    mat_arr.append(cv2.imread("./sample/"+path))

padded = pad_images_to_same_size(mat_arr)
for im in padded:
    cv2.imshow("padded", im)
    cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()