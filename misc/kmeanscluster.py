import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('kmc_test1.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plt.subplot(141), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("Input")

Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
compactness, label, center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
plt.subplot(142), plt.imshow(cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)), plt.title("K={}".format(K))

# res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)
# cv2.imshow('res2',res2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


""" Additional Ks """
# define criteria, number of clusters(K) and apply kmeans()
K = 5
compactness, label, center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
plt.subplot(143), plt.imshow(cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)), plt.title("K={}".format(K))

K = 10
compactness, label, center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
plt.subplot(144), plt.imshow(cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)), plt.title("K={}".format(K))

plt.show()