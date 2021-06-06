import cv2
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
startTime = datetime.now()

def find_nearest_white(nonzero, target):
    distances = np.sqrt((nonzero[:,:,0] - target[0]) ** 2 + (nonzero[:,:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)

    # print("near_index:", nearest_index)
    # print("elem:", distances[nearest_index])
    
    return nonzero[nearest_index]

old_src = cv2.imread("GOPR0250.JPG")
new_src = cv2.imread("new_coral1.jpg")

old_pink = cv2.imread("eyedrop_pink_mask.JPG")
old_white = cv2.imread("eyedrop_white_mask.jpg")
new_pink = cv2.imread("new_pink_mask.jpg")
new_white = cv2.imread("new_white_mask.jpg")

new_pink = cv2.erode(new_pink, np.ones((2,2),dtype=np.uint8))

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

beforeloop = datetime.now()
print("time to bef loop:", beforeloop - startTime)
for coor in new_nonzero:
    new_pink_copy = new_pink.copy()
    # cv2.circle(new_pink_copy, (coor[0][0], coor[0][1]), 5, (0,0,255), -1)
    # cv2.imshow("ITER", new_pink_copy)
    
    # nearest_px = find_nearest_white(ref_nonzero, coor[0])
    
    target = coor[0]
    x, y = target[0], target[1]
    # print("target:", target)

    # lower_x = x - 30
    # upper_x = x + 30
    # lower_y = y - 30
    # upper_y = y + 30

    # if lower_x < 0: x = 0
    # if lower_y < 0: y = 0
    # if upper_x > width:  upper_x = width
    # if upper_y > height: upper_x = width

    distances = np.sqrt((ref_nonzero[:,:,0] - x) ** 2 + (ref_nonzero[:,:,1] - y) ** 2)
    # print("len(dist):", len(distances))
    nearest_index = np.argmin(distances)
    min_distance = distances[nearest_index][0]
    # print("min_distance", min_distance)

    if min_distance > 30:
        # print("enter draw canvas with min distance:", min_distance)
        # canvas[y, x] = 255
        # cv2.circle(canvas, (x,y), 5, (0,255,0), 1)
        canvas.itemset((y,x,1),255)
        # cv2.imshow("canvas", canvas)

    ref_pink_copy = old_pink.copy()
    # cv2.circle(ref_pink_copy, (x, y), 5, (255,0,0), -1)
    # cv2.imshow("ref_pink_copy", ref_pink_copy)

    # if cv2.waitKey(1) == 27:
    #     break
afterloop = datetime.now()
print("time in loop:", afterloop - beforeloop)

cv2.imshow("canvas", canvas)
# print("done")

canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
contours, hier = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = max(contours, key=cv2.contourArea)
cont_x, cont_y, cont_w, cont_h = cv2.boundingRect(c)

new_coral = cv2.imread("cropped.jpg")
cv2.rectangle(new_coral, (cont_x-10, cont_y), (cont_x+cont_w, cont_y+cont_h+35), (0,255,0), 5)
cv2.imshow("new_coral", new_coral)

endtime = datetime.now()
print("misc after loop:", endtime - afterloop)
print("total time:", endtime - startTime)

cv2.waitKey(0)
cv2.destroyAllWindows()

plt.subplot(131), plt.imshow(cv2.cvtColor(old_src, cv2.COLOR_BGR2RGB)), plt.title("Reference Coral")
plt.subplot(132), plt.imshow(cv2.cvtColor(new_src, cv2.COLOR_BGR2RGB)), plt.title("Present Coral")
plt.subplot(133), plt.imshow(cv2.cvtColor(new_coral, cv2.COLOR_BGR2RGB)), plt.title("Changes Detected")

# plt.tight_layout()
# plt.show()

print("end")