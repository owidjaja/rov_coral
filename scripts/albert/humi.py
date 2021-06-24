import cv2
from matplotlib import pyplot as plt



old = cv2.imread("old_src.jpg")
old_ROI = cv2.imread("old_cropped.jpg")

f1 = plt.figure(1)

plt.subplot(211), plt.imshow(cv2.cvtColor(old, cv2.COLOR_BGR2RGB)), plt.title("Reference Coral")
plt.axis('off')
plt.subplot(212), plt.imshow(cv2.cvtColor(old_ROI, cv2.COLOR_BGR2RGB)), plt.title("Reference ROI")
plt.axis('off')


def plotf(i, path, fontsize=16):
    f = plt.figure(i)

    new = cv2.imread("out/" + path + "/new_src.jpg")
    new_ROI = cv2.imread("out/" + path + "/new_cropped.jpg")

    new_pink_mask  = cv2.imread("out/" + path + "/new_pink.jpg")
    new_white_mask = cv2.imread("out/" + path + "/new_white.jpg")

    canvas    = cv2.imread("out/" + path + "/canvas.jpg")
    final_new = cv2.imread("out/" + path + "/new_drawn.jpg")

    plt.subplot(231), plt.imshow(cv2.cvtColor(new, cv2.COLOR_BGR2RGB)), plt.title("Current Coral", fontsize=fontsize)
    plt.axis('off')
    plt.subplot(234), plt.imshow(cv2.cvtColor(new_ROI, cv2.COLOR_BGR2RGB)), plt.title("Current ROI", fontsize=fontsize)
    plt.axis('off')

    plt.subplot(232), plt.imshow(cv2.cvtColor(new_pink_mask, cv2.COLOR_BGR2RGB)), plt.title("Current Pink Mask", fontsize=fontsize)
    plt.axis('off')
    plt.subplot(235), plt.imshow(cv2.cvtColor(new_white_mask, cv2.COLOR_BGR2RGB)), plt.title("Current White Mask", fontsize=fontsize)
    plt.axis('off')

    plt.subplot(233), plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)), plt.title("Canvas", fontsize=fontsize)
    plt.axis('off')
    plt.subplot(236), plt.imshow(cv2.cvtColor(final_new, cv2.COLOR_BGR2RGB)), plt.title("Output", fontsize=fontsize)
    plt.axis('off')

fontsize = 16
plotf(2, "out1", fontsize=fontsize)
plotf(3, "out2", fontsize=fontsize)


plt.tight_layout()

# left  = 0.125  # the left side of the subplots of the figure
# right = 0.9    # the right side of the subplots of the figure
# bottom = 0.1   # the bottom of the subplots of the figure
# top = 0.9      # the top of the subplots of the figure
# wspace = 0.2   # the amount of width reserved for blank space between subplots
# hspace = 0.2   # the amount of height reserved for white space between subplots
# plt.subplots_adjust()

plt.show()