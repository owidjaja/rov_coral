import cv2
import numpy as np

img = cv2.imread("coral_under3.jpg")
cv2.imshow("src", img)

mask = np.zeros((img.shape[0:2]), dtype=np.uint8)
print(mask.shape)
print(mask[0,0])
cv2.rectangle(mask, (200,200), (400,400), 3, -1)

cv2.imshow("mask", mask)
cv2.waitKey(0)

output = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("output", output)
cv2.waitKey(0)