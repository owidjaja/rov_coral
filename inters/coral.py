#!/usr/bin/python3 -u

import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from datetime import datetime

# PATH = "/home/hammerhead/everythingThatWeClone/coral_colony_health_task2.2/inters/sample/"
PATH = "C:/Users/oscar/OneDrive - HKUST Connect/Documents/school work/ROV/coral_colony_health_task2.2/inters/sample/"
RATIO = 0.50

GREEN  = (0,255,0)
BLUE   = (255,0,0)
YELLOW = (0,255,255)
RED    = (0,0,255)

MIN_AREA = 1000
MAX_DIST = 15
WAITKEY = 1

def auto_resize(img, target_width=800):
    # print("Original Dimension: ", img.shape)

    orig_height, orig_width = img.shape[:2]
    scale_ratio = target_width / orig_width

    new_width = int(img.shape[1] * (scale_ratio))
    new_height= int(img.shape[0] * (scale_ratio))
    dim = (new_height, new_width)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # print("Resized Dimension: ", resized.shape, '\n')
    return resized

def pad_images_to_same_size(images):
    """
    :param images: sequence of images
    :return: list of images padded so that all images have same width and height (max width and height are used)
    """
    width_max = 0
    height_max = 0
    for img in images:
        h, w = img.shape[:2]
        width_max = max(width_max, w)
        height_max = max(height_max, h)

    images_padded = []
    for img in images:
        h, w = img.shape[:2]

        diff_vert = height_max - h
        # pad_top = diff_vert//2
        # pad_bottom = diff_vert - pad_top
        pad_top = diff_vert
        pad_bottom = 0

        diff_hori = width_max - w
        pad_left = diff_hori//2
        pad_right = diff_hori - pad_left

        img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        assert img_padded.shape[:2] == (height_max, width_max)
        images_padded.append(img_padded)

    return images_padded

def get_nonzero(mask, to_open=False, ksize=3):
    ret, thresh = cv2.threshold(mask, 27, 255, cv2.THRESH_BINARY)

    if to_open is True:
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((ksize,ksize), np.uint8))
        # cv2.imshow("open thresh", thresh)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return cv2.findNonZero(thresh)

def process_changes(canvas1, canvas2, this_nonzero, other_pink_nonzero, other_white_nonzero, ispink, color1, color2, max_dist=30):
    counter = 0
    hasChange = True
    # cv2.imshow("canvas", canvas)
    cv2.waitKey(1)

    for coor in this_nonzero:
        counter += 1
        target = coor[0]
        tar_x, tar_y = target[0], target[1]
        can_h, can_w = canvas1.shape[:2]
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
            hasChange = True
            canvas1[tar_y, tar_x] = color1
            

        elif ispink==False and pink_distance < white_distance:
            """ BLEACHING (RED) """
            hasChange = True
            canvas2[tar_y, tar_x] = color2
            
        elif ispink==True and white_distance < pink_distance:
            """ RECOVERY (BLUE) """
            hasChange = True
            canvas2[tar_y, tar_x] = color2

        # if hasChange == True and counter > 10000:
        #     print("counter:", counter)
        #     hasChange = False
        #     counter = 0
        #     cv2.imshow("canvas", canvas)
        #     if cv2.waitKey(1) == 27:
        #         break

    return canvas1, canvas2


def get_diff(old_pink, old_white, new_pink, new_white, max_dist=30, close_ksize=0):
    oldpink_nonzero  = get_nonzero(old_pink)
    oldwhite_nonzero = get_nonzero(old_white, to_open=True)
    newpink_nonzero  = get_nonzero(new_pink)
    newwhite_nonzero = get_nonzero(new_white, to_open=True)

    height, width = new_pink.shape[:2]
    canvas_zero   = np.zeros((height, width, 3), dtype=np.uint8)
    canvas_green  = canvas_zero.copy()
    canvas_blue   = canvas_zero.copy()
    canvas_red    = canvas_zero.copy()
    canvas_yellow = canvas_zero.copy()

    print("pc for growth(green) or recovery(blue)")
    canvas_green, canvas_blue = process_changes(canvas_green, canvas_blue, newpink_nonzero, oldpink_nonzero, oldwhite_nonzero, ispink=True , color1=(0,255,0)  , color2=(255,0,0), max_dist=max_dist)
    cv2.imshow("canvas1", cv2.bitwise_or(canvas_green, canvas_blue))
    print("done")
    # cv2.waitKey(0)

    print("pc for growth(green) or bleaching(red)")
    canvas_green, canvas_red = process_changes(canvas_green, canvas_red, newwhite_nonzero, oldpink_nonzero, oldwhite_nonzero, ispink=False, color1=(0,255,0)  , color2=(0,0,255), max_dist=max_dist)
    cv2.imshow("canvas2", cv2.bitwise_or(canvas_green, canvas_red))
    print("done")
    # cv2.waitKey(0)

    print("pc for death(yellow) or bleaching(red)")
    canvas_yellow, canvas_red = process_changes(canvas_yellow, canvas_red, oldpink_nonzero, newpink_nonzero, newwhite_nonzero, ispink=True , color1=(0,255,255), color2=(0,0,255), max_dist=max_dist)
    cv2.imshow("canvas3", cv2.bitwise_or(canvas_yellow, canvas_red))
    print("done")
    # cv2.waitKey(0)

    print("pc for death(yellow) or recovery(blue)")
    canvas_yellow, canvas_blue = process_changes(canvas_yellow, canvas_blue, oldwhite_nonzero, newpink_nonzero, newwhite_nonzero, ispink=False, color1=(0,255,255), color2=(255,0,0), max_dist=max_dist)
    cv2.imshow("canvas4", cv2.bitwise_or(canvas_yellow, canvas_blue))
    print("done")
    # cv2.waitKey(0)

    if close_ksize > 0:
        kernel = np.ones((close_ksize,close_ksize), np.uint8)
        canvas_green = cv2.morphologyEx(canvas_green, cv2.MORPH_CLOSE, kernel)
        canvas_yellow = cv2.morphologyEx(canvas_yellow, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow("canvasBGR opened", canvas)

    cv2.destroyAllWindows()
    # cv2.imshow("canvas_green", canvas_green)
    # cv2.imshow("canvas_blue", canvas_blue)
    # cv2.imshow("canvas_red", canvas_red)
    # cv2.imshow("canvas_yellow", canvas_yellow)
    # cv2.waitKey(0)

    return canvas_green, canvas_blue, canvas_red, canvas_yellow


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
            color = YELLOW
        elif cb == max_bgr:
            color = BLUE
        elif cg == max_bgr:
            color = GREEN
        else:
            color = RED

        cont_x, cont_y, cont_w, cont_h = cv2.boundingRect(cont)
        cv2.rectangle(new_coral, (cont_x-10, cont_y-5), (cont_x+cont_w+5, cont_y+cont_h+20), color, 5)

    return new_coral

def draw_diff_from_multiple_canvas(canvas_arr, new_cropped, min_area=MIN_AREA):
    new_coral = new_cropped
    color_arr = [GREEN, BLUE, RED, YELLOW]
    i = -1
    cont_drawn = 0
    for canvas in canvas_arr:
        i += 1
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        contours_ret = cv2.findContours(canvas_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print("canvas len(contours):", len(contours))

        contours = imutils.grab_contours(contours_ret)
        for cont in contours:
            area = cv2.contourArea(cont)
            if area < min_area:
                continue

            print("Drawing Contour with Area:", area)
            cont_x, cont_y, cont_w, cont_h = cv2.boundingRect(cont)
            cv2.rectangle(new_coral, (cont_x-10, cont_y-5), (cont_x+cont_w+5, cont_y+cont_h+20), color_arr[i], 5)

            cont_drawn += 1

            """ Omitting for now since sample randomized image has 5 changes, expected 4 """
            # if cont_drawn == 4:
            #     print("Drawn 4 rectangles, exiting draw function")
                # return new_coral
        
    return new_coral

def amplify_contours(canvas, min_area=MIN_AREA//2, dilate_ksize=5, close_ksize=3):
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    contours_ret = cv2.findContours(canvas_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("canvas len(contours):", len(contours))

    contours = imutils.grab_contours(contours_ret)
    for cont in contours:
        area = cv2.contourArea(cont)
        # print("Contour Area:", area)
        if area < min_area:
            cont_x, cont_y, cont_w, cont_h = cv2.boundingRect(cont)
            cv2.rectangle(canvas, (cont_x, cont_y), (cont_x+cont_w, cont_y+cont_h), (0,0,0), -1)

    cv2.imshow("filtered contours", canvas)
    cv2.waitKey(0)

    canvas = cv2.dilate(canvas, np.ones((dilate_ksize,dilate_ksize), dtype=np.uint8))
    cv2.imshow("amplified contours", canvas)
    cv2.waitKey(0)

    return canvas


########## MAIN ##########

src_arr = [ 'coral_ref.jpg',
            'coral-colony-test-1_51268948073_o.jpg',
            'coral-colony-test-2_51268762126_o.jpg',
            'coral-colony-test-3_51269790240_o.jpg',
            'Coral Colony F.png']

if __name__ == "__main__":
    print("Growth     GREEN")
    print("Recovery   BLUE")
    print("Bleaching  RED")
    print("Death      YELLOW")

    # cv2.namedWindow("src", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("old_pink", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("old_white", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("new_pink", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("new_white", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("canvas", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("new_drawn", cv2.WINDOW_NORMAL)

    old = cv2.imread(PATH + src_arr[0])
    new = cv2.imread(PATH + src_arr[4])

    if old is None or new is None:
        exit("ERROR: failed to read image")

    old, new = pad_images_to_same_size([old, new])

    old = cv2.resize(old, ( int(old.shape[1]*RATIO), int(old.shape[0]*RATIO) ), interpolation=cv2.INTER_AREA)
    cv2.imshow("old", old)
    old_hsv = cv2.cvtColor(old, cv2.COLOR_BGR2HSV)

    new = cv2.resize(new, ( int(new.shape[1]*RATIO), int(new.shape[0]*RATIO) ), interpolation=cv2.INTER_AREA)
    cv2.imshow("new", new)
    new_hsv = cv2.cvtColor(new, cv2.COLOR_BGR2HSV)

    # cv2.waitKey(0)
    begin = datetime.now()

    lower_pink_hsv = (149,29,152)
    upper_pink_hsv = (180,182,255)

    lower_white_hsv = (76,0,154)
    upper_white_hsv = (118,77,255)

    old_pink  = cv2.inRange(old_hsv, lowerb=lower_pink_hsv, upperb=upper_pink_hsv)
    old_white = cv2.inRange(old_hsv, lowerb=lower_white_hsv, upperb=upper_white_hsv)
    # cv2.imshow("old_pink" , old_pink)
    # cv2.imshow("old_white", old_white)

    new_pink  = cv2.inRange(new_hsv, lowerb=lower_pink_hsv, upperb=upper_pink_hsv)
    new_white = cv2.inRange(new_hsv, lowerb=lower_white_hsv, upperb=upper_white_hsv)
    # cv2.imshow("new_pink" , new_pink)
    # cv2.imshow("new_white", new_white)

    height, width = new_pink.shape[:2]
    print("new_pink (h,w):", height, width)

    # cv2.waitKey(500)
    canvas_green, canvas_blue, canvas_red, canvas_yellow = \
        get_diff(old_pink, old_white, new_pink, new_white, max_dist=MAX_DIST, close_ksize=3)
    
    """ Special for canvas_green: remove bottom 3/8 for any potential growth """
    cg_h, cg_w = canvas_green.shape[:2]
    cv2.rectangle(canvas_green, (0,int(5/8*cg_h)), (cg_w,cg_h), (0,0,0), thickness=-1)
    
    # kernel = np.ones((5,5), np.uint8)
    # canvas_green = cv2.morphologyEx(canvas_green, cv2.MORPH_CLOSE, kernel)
    # canvas_yellow = cv2.morphologyEx(canvas_yellow, cv2.MORPH_CLOSE, kernel)

    # canvas_green = amplify_contours(canvas_green)

    # Only for imshow purposes, drawing diff is based on seperate canvases
    canvas = cv2.bitwise_or(canvas_red, canvas_blue)
    canvas = cv2.bitwise_or(canvas, canvas_yellow)
    canvas = cv2.bitwise_or(canvas, canvas_green)

    # print("esc to draw")
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("old", old)
    cv2.imshow("canvas", canvas)

    canvas_arr = [canvas_green, canvas_blue, canvas_red, canvas_yellow]
    # new_drawn = draw_diff(canvas, new, min_area=1000)
    new_drawn = draw_diff_from_multiple_canvas(canvas_arr, new, min_area=MIN_AREA)
    cv2.imshow("new_drawn", new_drawn)

    end = datetime.now()
    print("time:", end-begin)





plt.tight_layout()
# plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()