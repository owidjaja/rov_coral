import cv2
import numpy as np
from matplotlib import pyplot as plt

# reducing colors in an image
def kmc_color_quantization(src, K, bgr_or_hsv = 'b', plot_graph = False):
    if (bgr_or_hsv == 'b'):
        img = src
    elif (bgr_or_hsv == 'h'):
        img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    if (bgr_or_hsv == 'h'):
        res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)

    # TODO
    if (plot_graph == True):
        colors = []

        # Now separate the data, Note the flatten()
        for i in range(K):
            colors.append(Z[label.ravel()==i])

        # Plot the data
        plt.scatter(A[:,0],A[:,1])
        plt.scatter(B[:,0],B[:,1],c = 'r')
        plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
        plt.xlabel('Height'),plt.ylabel('Weight')
        plt.show()

    return res2
    

# main

img = cv2.imread("kmc_test1.jpg")
plt.subplot(141), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("Input")

K = 3
res1 = kmc_color_quantization(img, K, bgr_or_hsv='h')
plt.subplot(142), plt.imshow(cv2.cvtColor(res1, cv2.COLOR_BGR2RGB)), plt.title("K={}".format(K))

K = 5
res2 = kmc_color_quantization(img, K)
plt.subplot(143), plt.imshow(cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)), plt.title("K={}".format(K))

K = 10
res3 = kmc_color_quantization(img, K)
plt.subplot(144), plt.imshow(cv2.cvtColor(res3, cv2.COLOR_BGR2RGB)), plt.title("K={}".format(K))

plt.show()