import cv2
from matplotlib import pyplot as plt

old = cv2.imread("GOPR0250.jpg")
new = cv2.imread("new_coral1.jpg")

old_ROI = cv2.imread("cropped_old.jpg")
new_ROI = cv2.imread("cropped.jpg")

canvas = cv2.imread("canvas.jpg")
final_new = cv2.imread("final_new_coral.jpg")

plt.subplot(231), plt.imshow(cv2.cvtColor(old, cv2.COLOR_BGR2RGB)), plt.title("Reference Coral")
plt.axis('off')
plt.subplot(234), plt.imshow(cv2.cvtColor(new, cv2.COLOR_BGR2RGB)), plt.title("Present Coral")
plt.axis('off')

plt.subplot(232), plt.imshow(cv2.cvtColor(old_ROI, cv2.COLOR_BGR2RGB)), plt.title("Reference Coral ROI")
plt.axis('off')
plt.subplot(235), plt.imshow(cv2.cvtColor(new_ROI, cv2.COLOR_BGR2RGB)), plt.title("Present Coral ROI")
plt.axis('off')

plt.subplot(233), plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)), plt.title("Canvas for Changes")
plt.axis('off')
plt.subplot(236), plt.imshow(cv2.cvtColor(final_new, cv2.COLOR_BGR2RGB)), plt.title("Changes Detected")
plt.axis('off')

plt.tight_layout()

left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots
# plt.subplots_adjust()

plt.show()