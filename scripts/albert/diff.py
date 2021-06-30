import cv2
import numpy as np

def concat_vh(list_2d):
    """ 
        define a function for vertically concatenating images of the same size  and horizontally 
        https://www.geeksforgeeks.org/concatenate-images-using-opencv-in-python/
    """    
    return cv2.vconcat([cv2.hconcat(list_h) for list_h in list_2d])

def get_blocks(img, desc):
    height, width = img.shape[:2]

    # Define the window size
    windowsize_r = 50
    windowsize_c = 50

    n_rows = int(np.ceil((img.shape[0] - windowsize_r) / windowsize_r))
    n_cols = int(np.ceil((img.shape[1] - windowsize_c) / windowsize_c))
    print("row, col:", n_rows, n_cols)

    window = np.zeros((windowsize_r, windowsize_c), dtype=np.uint8)
    blocks = [ [ window for j in range (n_cols)] for i in range (n_rows)]

    cv2.namedWindow("window")
    for r in range(0, img.shape[0] - windowsize_r, windowsize_r):
        for c in range(0, img.shape[1] - windowsize_c, windowsize_c):
            window = img[r:r+windowsize_r, c:c+windowsize_c]

            white_pixels = np.sum(window > 0)
            if white_pixels > (windowsize_r * windowsize_c // 4):
                cv2.circle(window, (25,25), 5, (0,255,0), -1)
            else:
                cv2.circle(window, (25,25), 5, (0,0,255), -1)

            blocks[r//windowsize_r][c//windowsize_c] = window

            # cv2.imshow("window", window)

            # k = cv2.waitKey(1)
            # if k == 27:
            #     exit("EXIT: Keyboard Interrupt")

    return blocks


old_pink = cv2.imread("old_pink_mask.JPG")
old_white = cv2.imread("old_white_mask.jpg")
new_pink = cv2.imread("new_pink_mask.jpg")
new_white = cv2.imread("new_white_mask.jpg")

old_pink_blocks = get_blocks(old_pink, "op")
old_white_blocks = get_blocks(old_white, "ow")
new_pink_blocks = get_blocks(new_pink, "np")
new_whote_blocks = get_blocks(new_white, "nw")

res = concat_vh(old_pink_blocks)

cv2.namedWindow("res", cv2.WINDOW_NORMAL)
cv2.imshow("res", res)
# res = cv2.resize(res, ( int(res.shape[1]*0.15), int(res.shape[0]*0.15) ), interpolation=cv2.INTER_AREA)
cv2.waitKey(0)