import cv2
import numpy as np

def auto_resize(img, target_width=800):
    print("Original Dimension: ", img.shape)

    orig_height, orig_width = img.shape[:2]
    scale_ratio = target_width / orig_width

    new_width = int(img.shape[1] * (scale_ratio))
    new_height= int(img.shape[0] * (scale_ratio))
    dim = (new_width, new_height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print("Resized Dimension: ", resized.shape, '\n')
    return resized

old = cv2.imread("eyedrop_pink_mask.jpg")
new = cv2.imread("new_pink_mask.jpg")

new = cv2.resize(new, (360, 409))

print(old.shape)
print(new.shape)

# old = auto_resize(old)
# new = auto_resize(new)

cv2.imshow("old", old)
cv2.imshow("new", new)

diff = cv2.bitwise_xor(old, new)
cv2.imshow("diff", diff)


cv2.waitKey(0)