import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("reference_coral_mask.jpg")
cv2.rectangle(img, (182,0), (382,345), (0,0,0), -1)
cv2.rectangle(img, (520,0), (700,240), (0,0,0), -1)


plt.imshow(img)

plt.show()

cv2.imwrite("coral_mid_right_gone.jpg", img)