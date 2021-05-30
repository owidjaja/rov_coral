import cv2
import numpy as np

""" take in input mask, and edit values onto mask to simulate grabcut rect mask """

img_src = cv2.imread("coral_under3.jpg")
mask = cv2.imread("coral_mask.jpg", cv2.IMREAD_GRAYSCALE)
# mask = cv2.imread("coral_mask.jpg", cv2.IMREAD_COLOR)

# print(mask, "\n")
# print(mask[0], "\n")
# print(mask[0][0], "\n")
print("mask.shape", mask.shape)
height, width = mask.shape
cv2.waitKey(0)

# cv2.imshow("BEFORE THRESHOLD", mask)
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow("thresh", mask)
# cv2.waitKey(0)


# for i in range(0, height):
#     for j in range(0, width):
#         if np.any(mask[i,j] != 0):      # green for probable foreground
#             img_src = cv2.circle(img_src, (j-1, i-1), radius=1, color=[0,255,0], thickness= -1)
#             mask = cv2.circle(mask, (j-1, i-1), 1, 3, -1)
#         else:                           # red for probable background
#             img_src = cv2.circle(img_src, (j-1, i-1), 2, [0,0,255], -1)
#             mask = cv2.circle(mask, (j-1, i-1), 2, 2, -1)

# for i in range(0, height):
#     for j in range(0, width):
#         if np.any(mask[i,j] != 0):      # green for probable foreground
#             img_src[i, j] = (0,255,0)
#             mask[i, j] = 3
#         else:
#             img_src[i, j] = (0,0,255)
#             mask[i, j] = 2

# colored_pixels = np.where(
#     (mask[:, :, 0] != 0) &
#     (mask[:, :, 1] != 0) & 
#     (mask[:, :, 2] != 0)
# )

# get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
black_pixels = np.where(
    (mask[:, :] == 0)
)

colored_pixels = np.where(
    (mask[:, :] != 0)
)

img_src[black_pixels] = (0,0,255)
mask[black_pixels] = 2

img_src[colored_pixels] = (0,255,0)
mask[colored_pixels] = 3

_, mask = cv2.threshold(mask, 2, 255, cv2.THRESH_BINARY)

cv2.imshow("img_src", img_src)
cv2.imshow("mask", mask)

cv2.waitKey(0)
cv2.imwrite("edited_src.jpg", img_src)
cv2.imwrite("edited_mask.jpg", mask)