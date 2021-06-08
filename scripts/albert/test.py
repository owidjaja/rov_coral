import cv2
import numpy as np

# pink  = cv2.imread("out/new_pink.jpg")
# white = cv2.imread("out/old_white.jpg")

# height, width = white.shape[:2]
# pink = cv2.resize(pink, (width, height))

# print(pink.shape)
# print(white.shape)

# cv2.imshow("pink", pink)
# cv2.imshow("white", white)

# comb = cv2.bitwise_or(pink, white)
# cv2.imshow("comb", comb)

canvas = cv2.imread("out/canvas.jpg")
cv2.imshow("canvas", canvas)

ksize = 5
kernel = np.ones((ksize,ksize), dtype=np.uint8)
closed = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
cv2.imshow("closed", closed)

cv2.waitKey(0)