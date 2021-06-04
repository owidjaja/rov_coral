import cv2
import numpy as np

def find_nearest_white(nonzero, target):
    distances = np.sqrt((nonzero[:,:,0] - target[0]) ** 2 + (nonzero[:,:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)

    # print("near_index:", nearest_index)
    # print("elem:", distances[nearest_index])
    
    return nonzero[nearest_index]

old_pink = cv2.imread("eyedrop_pink_mask.JPG")
old_white = cv2.imread("eyedrop_white_mask.jpg")
new_pink = cv2.imread("new_pink_mask.jpg")
new_white = cv2.imread("new_white_mask.jpg")

temp = new_pink
new_pink = old_pink
old_pink = temp

gray = cv2.cvtColor(new_pink, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 27, 255, cv2.THRESH_BINARY)
# cv2.imshow("new_pink_thresh", thresh)
new_nonzero = cv2.findNonZero(thresh)

gray = cv2.cvtColor(old_pink, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 27, 255, cv2.THRESH_BINARY)
# cv2.imshow("ref_pink_thresh", thresh)
ref_nonzero = cv2.findNonZero(thresh)

height, width = new_pink.shape[:2]
canvas = np.zeros((height, width, 3), dtype=np.uint8)

for coor in new_nonzero:
    new_pink_copy = new_pink.copy()
    cv2.circle(new_pink_copy, (coor[0][0], coor[0][1]), 5, (0,0,255), -1)
    cv2.imshow("ITER", new_pink_copy)
    
    # nearest_px = find_nearest_white(ref_nonzero, coor[0])
    
    target = coor[0]
    x, y = target[0], target[1]
    # print("target:", target)

    distances = np.sqrt((ref_nonzero[:,:,0] - x) ** 2 + (ref_nonzero[:,:,1] - y) ** 2)
    nearest_index = np.argmin(distances)
    min_distance = distances[nearest_index][0]
    # print("min_distance", min_distance)

    if min_distance > 15:
        # print("enter draw canvas with min distance:", min_distance)
        # canvas[y, x] = 255
        cv2.circle(canvas, (x,y), 5, (0,255,0), 1)
        cv2.imshow("canvas", canvas)

    ref_pink_copy = old_pink.copy()
    cv2.circle(ref_pink_copy, (x, y), 5, (255,0,0), -1)
    cv2.imshow("ref_pink_copy", ref_pink_copy)

    # if cv2.waitKey(1) == 27:
    #     break
    
cv2.imshow("canvas", canvas)
print("done")
cv2.waitKey(0)