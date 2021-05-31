import cv2
import numpy as np

pink_mask = cv2.imread("eyedrop_pink_mask.jpg", cv2.IMREAD_GRAYSCALE)
grabcut_mask = cv2.imread("grabcut_mask.jpg", cv2.IMREAD_GRAYSCALE)

ret, pink_mask = cv2.threshold(pink_mask, 63, 255, cv2.THRESH_BINARY)
ret, grabcut_mask = cv2.threshold(grabcut_mask, 63, 255, cv2.THRESH_BINARY)

print(pink_mask.shape)
print(grabcut_mask.shape)

combined_mask = cv2.bitwise_and(grabcut_mask, grabcut_mask, mask=pink_mask)

cv2.imshow("pink_mask", pink_mask)
cv2.imshow("grabcut_mask", grabcut_mask)
cv2.imshow("combined_mask", combined_mask)

cv2.waitKey(0)