import cv2
import numpy as np
from numpy.core.defchararray import upper

old_pink = cv2.imread("old_pink_mask.jpg")
old_white = cv2.imread("old_white_mask.jpg")
new_pink = cv2.imread("new_pink_mask.jpg")
new_white = cv2.imread("new_white_mask.jpg")

# ret, test_image = cv2.threshold(old_pink, 32, 255, cv2.THRESH_BINARY)
test_image = cv2.imread("new_coral1.jpg")
test_image = cv2.resize(test_image, ( int(test_image.shape[1]*0.15), int(test_image.shape[0]*0.15) ), interpolation=cv2.INTER_AREA)
# ret, test_image = cv2.threshold(old_pink, 32, 255, cv2.THRESH_BINARY)
cv2.imshow("test_image", test_image)
print("test_image.shape:", test_image.shape)
cv2.waitKey(0)

# Define the window size
windowsize_r = 50
windowsize_c = 50

cv2.namedWindow("window")

n_rows = int(np.ceil((test_image.shape[0] - windowsize_r) / windowsize_r))
n_cols = int(np.ceil((test_image.shape[1] - windowsize_c) / windowsize_c))
print(n_rows, n_cols)

window = np.zeros((50,50), dtype=np.uint8)

blocks = [ [ window for j in range (n_cols)] for i in range (n_rows)]

count_row = 0
count_col = 0

for r in range(0, test_image.shape[0] - windowsize_r, windowsize_r):
    for c in range(0, test_image.shape[1] - windowsize_c, windowsize_c):
        window = test_image[r:r+windowsize_r, c:c+windowsize_c]

        # white_pixels = np.sum(window > 0)
        # if white_pixels > (windowsize_r * windowsize_c // 4):
        #     cv2.circle(window, (40,10), 5, (0,255,0), -1)
        # else:
        #     cv2.circle(window, (40,10), 5, (0,0,255), -1)

        blocks[r//windowsize_r][c//windowsize_c] = window

        cv2.imshow("window", window)

        k = cv2.waitKey(1)
        if k == 27:
            exit("EXIT: Keyboard Interrupt")
        
        # count_col += 1
        
    # print(count_col)
    # count_col = 0
    # count_row += 1

print(count_row)
# print(blocks)

def concat_vh(list_2d):
    """ 
        define a function for vertically concatenating images of the same size  and horizontally 
        https://www.geeksforgeeks.org/concatenate-images-using-opencv-in-python/
    """    
    return cv2.vconcat([cv2.hconcat(list_h) for list_h in list_2d])

res = concat_vh(blocks)

cv2.imshow("res", res)
cv2.waitKey(0)