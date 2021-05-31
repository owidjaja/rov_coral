import cv2
import numpy as np
import matplotlib.pyplot as plt

image_original = cv2.imread('../res/coral_under3.jpg', cv2.IMREAD_COLOR)
# remove noise
image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
# Reduce noise in image
img = cv2.GaussianBlur(image_gray,(3,3),0)
# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)
filtered_image = cv2.Laplacian(img, ksize=3, ddepth=cv2.CV_16S)
# converting back to uint8
filtered_image = cv2.convertScaleAbs(filtered_image)

# Plot outputs
(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 15))
ax1.title.set_text('Original Image')
ax1.imshow(image_original)

ax2.title.set_text('Laplacian Filtered Image')
ax2.imshow(filtered_image, cmap='gray')

plt.show()