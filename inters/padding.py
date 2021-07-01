#!/usr/bin/env python
import cv2

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
        # pad_top = diff_vert//2
        # pad_bottom = diff_vert - pad_top
        pad_top = diff_vert
        pad_bottom = 0

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

count = 0
mat_arr = []
for path in src_arr:
    im = auto_resize(cv2.imread("./sample/"+path), target_width=600)
    mat_arr.append(im)

    print("{}:{}".format(count, im.shape))
    count += 1

print("")
count = 0
padded = pad_images_to_same_size(mat_arr)
for im in padded:
    cv2.imshow("padded", im)
    
    print("{}:{}".format(count, im.shape))
    count += 1

    cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()