import cv2
import numpy as np
from datetime import datetime
startTime = datetime.now()

old_pink = cv2.imread("eyedrop_pink_mask.JPG")
old_white = cv2.imread("eyedrop_white_mask.jpg")
new_pink = cv2.imread("new_pink_mask.jpg")
new_white = cv2.imread("new_white_mask.jpg")

def get_nonzero(img, to_open=False, ksize=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 27, 255, cv2.THRESH_BINARY)

    if to_open is True:
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((ksize,ksize), np.uint8))
        # cv2.imshow("open thresh", thresh)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return cv2.findNonZero(thresh)

""" PAST CORAL: OLD PINK """
# gray = cv2.cvtColor(old_pink, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 27, 255, cv2.THRESH_BINARY)
# cv2.imshow("ref_pink_thresh", thresh)
# oldpink_nonzero = cv2.findNonZero(thresh)
oldpink_nonzero = get_nonzero(old_pink)

""" PAST CORAL: OLD WHITE """
# gray = cv2.cvtColor(old_white, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 27, 255, cv2.THRESH_BINARY)
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
# cv2.imshow("ref_white_thresh", thresh)
# oldwhite_nonzero = cv2.findNonZero(thresh)
oldwhite_nonzero = get_nonzero(old_white, to_open=True)

""" CURRENT CORAL: NEW PINK """
# gray = cv2.cvtColor(new_pink, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 27, 255, cv2.THRESH_BINARY)
# cv2.imshow("new_pink_thresh", thresh)
# newpink_nonzero = cv2.findNonZero(thresh)
newpink_nonzero = get_nonzero(new_pink)

""" CURRENT CORAL: NEW WHITE """
newwhite_nonzero = get_nonzero(new_white, to_open=True)

cv2.waitKey(0)
# cv2.destroyAllWindows()

height, width = new_pink.shape[:2]
canvas = np.zeros((height, width, 3), dtype=np.uint8)
MAX_DIST = 30

beforeloop = datetime.now()
print("time to bef loop:", beforeloop - startTime)
for coor in newpink_nonzero:
    inloop = datetime.now()

    target = coor[0]
    tar_x, tar_y = target[0], target[1]

    # new_pink_copy = new_pink.copy()
    # cv2.circle(new_pink_copy, (tar_x, tar_y), 5, (0,0,255), -1)
    # cv2.imshow("ITER", new_pink_copy)

    befdist = datetime.now()
    # print("before disatnce:", befdist - inloop)

    oldpink_distances = np.sqrt((oldpink_nonzero[:,:,0] - tar_x) ** 2 + (oldpink_nonzero[:,:,1] - tar_y) ** 2)
    nearest_index = np.argmin(oldpink_distances)
    pink_distance = oldpink_distances[nearest_index][0]
    
    oldwhite_distances = np.sqrt((oldwhite_nonzero[:,:,0] - tar_x) ** 2 + (oldwhite_nonzero[:,:,1] - tar_y) ** 2)
    nearest_index = np.argmin(oldwhite_distances)
    white_distance = oldwhite_distances[nearest_index][0]
    
    aftdist = datetime.now()
    # print("aft dsit:", aftdist - befdist)

    if white_distance > MAX_DIST and pink_distance > MAX_DIST:
        """ GROWTH (GREEN) """
        # canvas[y, x] = 255
        # cv2.circle(canvas, (tar_x, tar_y), 5, (0,255,0), 1)
        canvas.itemset((tar_y, tar_x, 1), 255)
    elif white_distance < pink_distance:
        """ RECOVERY (BLUE) """
        # cv2.circle(canvas, (tar_x, tar_y), 5, (255,0,0), 1)
        canvas.itemset((tar_y, tar_x, 0), 255)
    else:
        pass

    cv2.imshow("canvas", canvas)

    # ref_pink_copy = old_pink.copy()
    # cv2.circle(ref_pink_copy, (tar_x, tar_y), 5, (255,0,0), -1)
    # cv2.imshow("ref_pink_copy", ref_pink_copy)

    # if cv2.waitKey(1) == 27:
    #     break

    befexit = datetime.now()
    # print("misc after dist:", befexit - aftdist)
    # print("total inloop:", befexit - inloop)
    # exit()

for coor in newwhite_nonzero:
    target = coor[0]
    tar_x, tar_y = target[0], target[1]

    oldpink_distances = np.sqrt((oldpink_nonzero[:,:,0] - tar_x) ** 2 + (oldpink_nonzero[:,:,1] - tar_y) ** 2)
    nearest_index = np.argmin(oldpink_distances)
    pink_distance = oldpink_distances[nearest_index][0]
    
    oldwhite_distances = np.sqrt((oldwhite_nonzero[:,:,0] - tar_x) ** 2 + (oldwhite_nonzero[:,:,1] - tar_y) ** 2)
    nearest_index = np.argmin(oldwhite_distances)
    white_distance = oldwhite_distances[nearest_index][0]

    if white_distance > MAX_DIST and pink_distance > MAX_DIST:
        """ GROWTH (GREEN) """
        # cv2.circle(canvas, (tar_x, tar_y), 5, (0,255,0), 1)
        canvas.itemset((tar_y, tar_x, 1), 255)
    elif pink_distance < white_distance:
        """ BLEACHING (RED) """
        # cv2.circle(canvas, (tar_x, tar_y), 5, (0,0,255), 1)
        canvas.itemset((tar_y, tar_x, 2), 255)
    else:
        pass

    cv2.imshow("canvas", canvas)

    # if cv2.waitKey(1) == 27:
    #     break


afterloop = datetime.now()
print("time in loop:", afterloop - beforeloop)
cv2.destroyAllWindows()

cv2.imshow("canvas", canvas)
canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((10,10), np.uint8))
cv2.imshow("canvasBGR opened", canvas)
canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

# canvas_gray = cv2.morphologyEx(canvas_gray, cv2.MORPH_CLOSE, np.ones((10,10), np.uint8))
# cv2.imshow("canvas opened", canvas_gray)
# canvas_gray = cv2.morphologyEx(canvas_gray, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
# cv2.imshow("canvas after morph", canvas_gray)

contours, hier = cv2.findContours(canvas_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("len(contours):", len(contours))

new_coral = cv2.imread("cropped.jpg")
canvas_mask = np.zeros(canvas_gray.shape, np.uint8) # only need grayscale
MIN_AREA = 200
for cont in contours:
    area = cv2.contourArea(cont)
    # print(area)
    if area < MIN_AREA:
        continue

    cv2.drawContours(canvas_mask, cont, -1, 255, -1)
    cont_x, cont_y, cont_w, cont_h = cv2.boundingRect(cont)
    
    mean_b, mean_g, mean_r = cv2.mean(canvas, mask=canvas_mask)[:3]
    mean_b, mean_g, mean_r = int(mean_b), int(mean_g), int(mean_r)
    print("meanBGR:", mean_b, mean_g, mean_r)
    
    cv2.rectangle(new_coral, (cont_x-10, cont_y), (cont_x+cont_w, cont_y+cont_h+35), (mean_b,mean_g,mean_r), 5)

    cv2.imshow("new_coral", new_coral)
    cv2.waitKey(0)

cv2.imshow("FINAL new_coral", new_coral)

endtime = datetime.now()
print("misc after loop:", endtime - afterloop)
print("total time:", endtime - startTime)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("end")