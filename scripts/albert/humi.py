import cv2
from matplotlib import pyplot as plt

path = "out2"

old = cv2.imread("old_src.jpg")
new = cv2.imread("out/" + path + "/new_src.jpg")

old_ROI = cv2.imread("out/" + path + "/old_cropped.jpg")
new_ROI = cv2.imread("out/" + path + "/new_cropped.jpg")

new_pink_mask  = cv2.imread("out/" + path + "/new_pink.jpg")
new_white_mask = cv2.imread("out/" + path + "/new_white.jpg")

canvas    = cv2.imread("out/" + path + "/canvas.jpg")
final_new = cv2.imread("out/" + path + "/new_drawn.jpg")

plt.subplot(241), plt.imshow(cv2.cvtColor(old, cv2.COLOR_BGR2RGB)), plt.title("Reference Coral")
plt.axis('off')
plt.subplot(245), plt.imshow(cv2.cvtColor(new, cv2.COLOR_BGR2RGB)), plt.title("Current Coral")
plt.axis('off')

plt.subplot(242), plt.imshow(cv2.cvtColor(old_ROI, cv2.COLOR_BGR2RGB)), plt.title("Reference ROI")
plt.axis('off')
plt.subplot(246), plt.imshow(cv2.cvtColor(new_ROI, cv2.COLOR_BGR2RGB)), plt.title("Current ROI")
plt.axis('off')

plt.subplot(243), plt.imshow(cv2.cvtColor(new_pink_mask, cv2.COLOR_BGR2RGB)), plt.title("Current Pink Mask")
plt.axis('off')
plt.subplot(247), plt.imshow(cv2.cvtColor(new_white_mask, cv2.COLOR_BGR2RGB)), plt.title("Current White Mask")
plt.axis('off')

plt.subplot(244), plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)), plt.title("Canvas for Changes Detected")
plt.axis('off')
plt.subplot(248), plt.imshow(cv2.cvtColor(final_new, cv2.COLOR_BGR2RGB)), plt.title("Output")
plt.axis('off')

# plt.tight_layout()

left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots
# plt.subplots_adjust()

plt.show()