import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from datetime import datetime

def get_nonzero(mask, to_open=False, ksize=3):
    ret, thresh = cv2.threshold(mask, 27, 255, cv2.THRESH_BINARY)

    if to_open is True:
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((ksize,ksize), np.uint8))
        # cv2.imshow("open thresh", thresh)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return cv2.findNonZero(thresh)

def process_changes(canvas, this_nonzero, other_pink_nonzero, other_white_nonzero, ispink, color1, color2, max_dist=30):
    counter = 0
    hasChange = True
    cv2.imshow("canvas", canvas)
    cv2.waitKey(1)

    for coor in this_nonzero:
        counter += 1
        target = coor[0]
        tar_x, tar_y = target[0], target[1]
        can_h, can_w = canvas.shape[:2]
        # if (tar_x >= can_w) or (tar_y >= can_h): continue

        # out_pink = new_pink.copy()
        # cv2.circle(out_pink, (tar_x, tar_y), 5, (0,0,255), -1)
        # cv2.imshow("out_pink", out_pink)

        other_pink_distances = np.sqrt((other_pink_nonzero[:,:,0] - tar_x) ** 2 + (other_pink_nonzero[:,:,1] - tar_y) ** 2)
        nearest_index = np.argmin(other_pink_distances)
        pink_distance = other_pink_distances[nearest_index][0]
        
        other_white_distances = np.sqrt((other_white_nonzero[:,:,0] - tar_x) ** 2 + (other_white_nonzero[:,:,1] - tar_y) ** 2)
        nearest_index = np.argmin(other_white_distances)
        white_distance = other_white_distances[nearest_index][0]

        if white_distance > max_dist and pink_distance > max_dist:
            """ GROWTH (GREEN) OR DAMAGE (YELLOW) """
            canvas[tar_y, tar_x] = color1
            hasChange = True

        elif ispink==False and pink_distance < white_distance:
            """ BLEACHING (RED) """
            canvas[tar_y, tar_x] = color2
            hasChange = True
            
        elif ispink==True and white_distance < pink_distance:
            """ RECOVERY (BLUE) """
            canvas[tar_y, tar_x] = color2
            hasChange = True

        # if hasChange == True and counter > 10000:
        #     print("counter:", counter)
        #     hasChange = False
        #     counter = 0
        #     cv2.imshow("canvas", canvas)
        #     if cv2.waitKey(1) == 27:
        #         break

    return canvas

def get_diff(old_pink, old_white, new_pink, new_white, max_dist=30, close_ksize=0):
    oldpink_nonzero  = get_nonzero(old_pink)
    oldwhite_nonzero = get_nonzero(old_white, to_open=True)
    newpink_nonzero  = get_nonzero(new_pink)
    newwhite_nonzero = get_nonzero(new_white, to_open=True)

    height, width = new_pink.shape[:2]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # print("canvas.shape",canvas.shape)

    print("pc for growth or recovery")
    canvas = process_changes(canvas, newpink_nonzero , oldpink_nonzero, oldwhite_nonzero, ispink=True , color1=(0,255,0)  , color2=(255,0,0), max_dist=max_dist)
    cv2.imshow("canvas", canvas)
    # cv2.waitKey(0)
    print("pc for growth or bleaching")
    canvas = process_changes(canvas, newwhite_nonzero, oldpink_nonzero, oldwhite_nonzero, ispink=False, color1=(0,255,0)  , color2=(0,0,255), max_dist=max_dist)
    cv2.imshow("canvas", canvas)
    # cv2.waitKey(0)
    print("pc for death or bleaching")
    canvas = process_changes(canvas, oldpink_nonzero , newpink_nonzero, newwhite_nonzero, ispink=True , color1=(0,255,255), color2=(0,0,255), max_dist=max_dist)
    cv2.imshow("canvas", canvas)
    # cv2.waitKey(0)
    print("pc for death or recovery")
    canvas = process_changes(canvas, oldwhite_nonzero, newpink_nonzero, newwhite_nonzero, ispink=False, color1=(0,255,255), color2=(255,0,0), max_dist=max_dist)

    # cv2.imshow("canvas not opened", canvas)

    if close_ksize > 0:
        kernel = np.ones((close_ksize,close_ksize), np.uint8)
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow("canvasBGR opened", canvas)

    return canvas

def draw_diff(canvas, new_cropped, min_area=600):
    new_coral = new_cropped
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    contours_ret = cv2.findContours(canvas_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("canvas len(contours):", len(contours))

    # get top 4 contours by size
    contours = imutils.grab_contours(contours_ret)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

    for cont in contours:
        area = cv2.contourArea(cont)
        print("Contour area:", area)
        if area < min_area:
            continue

        x, y = cont[0][0]
        cont_bgr = canvas[y][x]
        cb, cg, cr = cont_bgr
        cb, cg, cr = int(cb), int(cg), int(cr)  # convert np.uint8 into python int

        max_bgr = max(cont_bgr)
        if abs(cg - cr) <= 15:
            color = (0,255,255)
        elif cb == max_bgr:
            color = (255,0,0)
        elif cg == max_bgr:
            color = (0,255,0)
        else:
            color = (0,0,255)

        cont_x, cont_y, cont_w, cont_h = cv2.boundingRect(cont)
        cv2.rectangle(new_coral, (cont_x-10, cont_y-5), (cont_x+cont_w+5, cont_y+cont_h+20), color, 5)

    return new_coral




def get_mask(hsv, lower, upper):
    return cv2.inRange(hsv, lowerb=lower, upperb=upper)

src_arr = ['coral-colony-test-1_51268948073_o.jpg','coral-colony-test-2_51268762126_o.jpg','coral-colony-test-3_51269790240_o.jpg','Coral Colony F.png']

if __name__ == "__main__":
    print("growth GREEN")
    print("recovery BLUE")
    print("bleaching RED")
    print("death YELLOW")

    # cv2.namedWindow("src", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("old_pink", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("old_white", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("new_pink", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("new_white", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("canvas", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("new_drawn", cv2.WINDOW_NORMAL)

    PATH = "/home/hammerhead/everythingThatWeClone/coral_colony_health_task2.2/inters/sample/"
    old = cv2.imread(PATH + src_arr[1])
    new = cv2.imread(PATH + src_arr[2])
    RATIO = 0.50

    if old is None or new is None:
        exit("ERROR: failed to read image")

    old = cv2.resize(old, ( int(old.shape[1]*RATIO), int(old.shape[0]*RATIO) ), interpolation=cv2.INTER_AREA)
    cv2.imshow("old", old)
    old_hsv = cv2.cvtColor(old, cv2.COLOR_BGR2HSV)

    new = cv2.resize(new, ( int(new.shape[1]*RATIO), int(new.shape[0]*RATIO) ), interpolation=cv2.INTER_AREA)
    # cv2.imshow("new", new)
    new_hsv = cv2.cvtColor(new, cv2.COLOR_BGR2HSV)

    # cv2.waitKey(0)
    begin = datetime.now()

    old_pink  = cv2.inRange(old_hsv, lowerb=(149,29,187), upperb=(180,149,255))
    old_white = cv2.inRange(old_hsv, lowerb=(76,0,154)  , upperb=(156,77,255))

    # cv2.imshow("old_pink" , old_pink)
    # cv2.imshow("old_white", old_white)

    new_pink  = cv2.inRange(new_hsv, lowerb=(149,29,187), upperb=(180,149,255))
    new_white = cv2.inRange(new_hsv, lowerb=(76,0,154)  , upperb=(156,77,255))

    # cv2.imshow("new_pink" , new_pink)
    # cv2.imshow("new_white", new_white)

    # pink_coral   = cv2.bitwise_and(src, src, mask=pink_mask)
    # white_coral  = cv2.bitwise_and(src, src, mask=white_mask)

    # cv2.imshow("pink_coral" , pink_coral)
    # cv2.imshow("white_coral", white_coral)

    # plt.subplot(131), plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)), plt.title("src")
    # plt.subplot(132), plt.imshow(cv2.cvtColor(pink_coral, cv2.COLOR_BGR2RGB)), plt.title("pink")
    # plt.subplot(133), plt.imshow(cv2.cvtColor(white_coral, cv2.COLOR_BGR2RGB)), plt.title("white")
    
    height, width = new_pink.shape[:2]
    print("new_pink (h,w):", height, width)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    cv2.waitKey(500)
    canvas = get_diff(old_pink, old_white, new_pink, new_white, max_dist=30, close_ksize=3)
    cv2.imshow("canvas", canvas)
    new_drawn = draw_diff(canvas, new, min_area=1000)
    cv2.imshow("new_drawn", new_drawn)
    end = datetime.now()
    print("time:", end-begin)





plt.tight_layout()
# plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
