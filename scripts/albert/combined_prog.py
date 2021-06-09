import cv2
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

""" align_base.py """

def auto_resize(img, target_width=800):
    # print("Original Dimension: ", img.shape)

    orig_height, orig_width = img.shape[:2]
    scale_ratio = target_width / orig_width

    new_width = int(img.shape[1] * (scale_ratio))
    new_height= int(img.shape[0] * (scale_ratio))
    dim = (new_width, new_height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # print("Resized Dimension: ", resized.shape, '\n')
    return resized

def extend_range(value, is_type_hue, upper_or_lower, tolerance):
    """ extend hsv range based on type_hue """

    if is_type_hue == 1:
        # set the boundary for hue (0-180)°
        boundary = 180
    else:
        # set the boundary for saturation and value (0-255)
        boundary = 255

    if upper_or_lower == 'u':
        if(value + tolerance > boundary):
            value = boundary
        else:
            value = value + tolerance
    
    elif upper_or_lower == 'l':
        if (value - tolerance < 0):
            value = 0
        else:
            value = value - tolerance

    return value

def get_black_base(hsv, actual_hsv=[105,198,18], tolerance=[30,50,30]):
    """ get black base mask to mask over img """

    # cv2.imshow("hsv", hsv)

    hue, sat, val = actual_hsv
    # print("Base hsv actual:", actual_hsv)

    hue_tol, sat_tol, val_tol = tolerance
    # print("Base hsv tolerance:", tolerance)

    hue_upper = extend_range(hue, 1, 'u', hue_tol)
    hue_lower = extend_range(hue, 1, 'l', hue_tol)
    sat_upper = extend_range(sat, 0, 'u', sat_tol)
    sat_lower = extend_range(sat, 0, 'l', sat_tol)
    val_upper = extend_range(val, 0, 'u', val_tol)
    val_lower = extend_range(val, 0, 'l', val_tol)

    upper =  np.array([hue_upper, sat_upper, val_upper])
    lower =  np.array([hue_lower, sat_lower, val_lower])
    # print("Base hsv range:", lower, upper, '\n')

    base_mask = cv2.inRange(hsv, lower, upper)
    # cv2.imshow("base_mask", base_mask)
    
    # cv2.waitKey(0)

    return base_mask

def get_mid_line(src, mask):
    """ close operation to remove noise, then draw line based on midpoint of biggest contour (black base) """

    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), dtype=np.uint8))

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    if num_contours == 0:
        print("ERROR: 0 contours found")
        cv2.imshow("closed", closed)
        cv2.waitKey(0)
        exit()
    else:
        # print("No. of contours found: ", num_contours)
        pass

    # find the biggest countour (c) by area
    c = max(contours, key = cv2.contourArea)

    x,y,w,h = cv2.boundingRect(c)
    # cv2.rectangle(src, (x,y), (x+w,y+h), (255,0,0), 2)

    height, width = src.shape[:2]
    # cv2.line(src, (x+w//2, 0), (x+w//2, height), (0,0,255), 2)

    # cv2.imshow("line", src)
    # cv2.waitKey(0)

    return [x, y, w, h]

def crop_to_standard(src, base_dim, approx_height_ratio=4.5, crop_extend=0.15, resize=True):
    x, y, w, h = base_dim

    lower_x = int(x - crop_extend * w)
    lower_y = int(y - (approx_height_ratio + crop_extend) * h)

    upper_x = int(x + (1 + crop_extend) * w)
    upper_y = int(y + (1 + crop_extend) * h)

    cropped = src[lower_y:upper_y, lower_x:upper_x]
    if resize is True:
        cropped = auto_resize(cropped, target_width=360)

    # cv2.imshow("cropped", cropped)
    # k = cv2.waitKey(0)
    # if k == ord('s'):
    #     print("saving cropped image with dim:", cropped.shape)
    #     cv2.imwrite("cropped_old.jpg", cropped)

    return cropped, (lower_x, lower_y, upper_x, upper_y)

def align_base(img, approx_height_ratio=4.5, crop_extend=0.15, resize=True):
    # src = auto_resize(src)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    base_mask = get_black_base(hsv)
    
    base_dim = get_mid_line(img, base_mask)
    
    return crop_to_standard(img, base_dim, approx_height_ratio=approx_height_ratio, crop_extend=crop_extend, resize=resize)









""" eyedropper.py """

def cb_nothing(x):
    # print(x)
    pass

def extend_range(value, is_type_hue, upper_or_lower, tolerance):
    # https://github.com/alieldinayman/HSV-Color-Picker/blob/master/HSV%20Color%20Picker.py
    # https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv

    """ Assign ranges based on hsv type and whether the range is upper or lower, with hardcoded tolerance value
        Still having trouble finding appropriate tolerance value for real underwater image """

    if is_type_hue == 1:
        # set the boundary for hue (0-180)°
        boundary = 180
    else:
        # set the boundary for saturation and value (0-255)
        boundary = 255

    if upper_or_lower == 'u':
        if(value + tolerance > boundary):
            value = boundary
        else:
            value = value + tolerance
    
    elif upper_or_lower == 'l':
        if (value - tolerance < 0):
            value = 0
        else:
            value = value - tolerance

    return value

def click_event(event, x, y, flags, param):
    # https://docs.opencv.org/master/db/d5b/tutorial_py_mouse_handling.html
    """ Left click on image in hsv window to inspect a pixel once
        Right click to inspect pixels as you move cursor """
    global adjusting, px_x, px_y

    img = param

    if event==cv2.EVENT_RBUTTONDOWN:
        adjusting = not adjusting

    elif event==cv2.EVENT_LBUTTONDOWN or (event==cv2.EVENT_MOUSEMOVE and adjusting==True):
        px_x, px_y = x, y
        hsv_val = img[y,x]
        # print("Actual HSV Values:", hsv_val)
        pass

def generate_mask(img, x, y):
    hsv_val = img[y,x]
    # print("Actual HSV Values:", hsv_val)
    hue = int(hsv_val[0])
    sat = int(hsv_val[1])
    val = int(hsv_val[2])

    HUE_TOLERANCE = cv2.getTrackbarPos("hue_track", "Trackbar_Window")
    SAT_TOLERANCE = cv2.getTrackbarPos("sat_track", "Trackbar_Window")
    VAL_TOLERANCE = cv2.getTrackbarPos("val_track", "Trackbar_Window")

    hue_upper = extend_range(hue, 1, 'u', HUE_TOLERANCE)
    hue_lower = extend_range(hue, 1, 'l', HUE_TOLERANCE)
    sat_upper = extend_range(sat, 0, 'u', SAT_TOLERANCE)
    sat_lower = extend_range(sat, 0, 'l', SAT_TOLERANCE)
    val_upper = extend_range(val, 0, 'u', VAL_TOLERANCE)
    val_lower = extend_range(val, 0, 'l', VAL_TOLERANCE)

    upper =  np.array([hue_upper, sat_upper, val_upper])
    lower =  np.array([hue_lower, sat_lower, val_lower])
    # print(lower, upper, '\n')

    return cv2.inRange(img,lower,upper)

def eyedrop(img):
    global px_x, px_y

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv", hsv)

    cv2.namedWindow("Trackbar_Window", cv2.WINDOW_NORMAL)

    print("Working on pink mask...")
    cv2.createTrackbar("hue_track", "Trackbar_Window", 6, 180//2, cb_nothing)
    cv2.createTrackbar("sat_track", "Trackbar_Window", 42, 255//2, cb_nothing)
    cv2.createTrackbar("val_track", "Trackbar_Window", 24, 255//2, cb_nothing)
    while (True):
        cv2.setMouseCallback('hsv', click_event, hsv)
        pink_mask = generate_mask(hsv, px_x, px_y)
        cv2.imshow("pink_mask", pink_mask)
        cv2.imshow("masked", cv2.bitwise_and(img, img, mask=pink_mask))
        if cv2.waitKey(1) == 27:
            break
        elif cv2.waitKey(1) == ord('s'):
            print("saving pink mask")
            # cv2.imwrite("eyedrop_pink_mask.jpg", pink_mask)
            break
    cv2.destroyWindow("pink_mask")

    print("Working on white mask...")
    cv2.createTrackbar("hue_track", "Trackbar_Window", 15, 180//2, cb_nothing)
    cv2.createTrackbar("sat_track", "Trackbar_Window", 18, 255//2, cb_nothing)
    cv2.createTrackbar("val_track", "Trackbar_Window", 10, 255//2, cb_nothing)
    while (True):
        cv2.setMouseCallback('hsv', click_event, hsv)
        white_mask = generate_mask(hsv, px_x, px_y)
        cv2.imshow("white_mask", white_mask)
        cv2.imshow("masked", cv2.bitwise_and(img, img, mask=white_mask))
        if cv2.waitKey(1) == 27:
            break
        elif cv2.waitKey(1) == ord('s'):
            print("saving white mask")
            # cv2.imwrite("eyedrop_white_mask.jpg", white_mask)
            break
    cv2.destroyWindow("white_mask")


    # if pink_mask is None or white_mask is None:
    #     exit("ERROR: mask empty")
    # pw_mask = cv2.bitwise_or(pink_mask, white_mask)
    # cv2.imshow("combined_mask", pw_mask)

    # k = cv2.waitKey(0)
    # if k == ord('m'):
    #     print("saving to coral_mask.jpg")
    #     cv2.imwrite("reference_coral_mask.jpg", pw_mask)
    # elif k == ord('s'):
    #     print("saving coral after mask as masked_perfect_coral.jpg")
    #     cv2.imwrite("masked_perfect_coral.jpg", cv2.bitwise_and(img,img,mask=pw_mask))
    
    cv2.destroyAllWindows()
    return pink_mask, white_mask





""" all_diff.py """

def get_nonzero(mask, to_open=False, ksize=3):
    ret, thresh = cv2.threshold(mask, 27, 255, cv2.THRESH_BINARY)

    if to_open is True:
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((ksize,ksize), np.uint8))
        # cv2.imshow("open thresh", thresh)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return cv2.findNonZero(thresh)

def process_changes(canvas, this_nonzero, other_pink_nonzero, other_white_nonzero, ispink, color1, color2, max_dist=30):
    for coor in this_nonzero:
        target = coor[0]
        tar_x, tar_y = target[0], target[1]
        can_h, can_w = canvas.shape[:2]
        # if (tar_x >= can_w) or (tar_y >= can_h): continue

        other_pink_distances = np.sqrt((other_pink_nonzero[:,:,0] - tar_x) ** 2 + (other_pink_nonzero[:,:,1] - tar_y) ** 2)
        nearest_index = np.argmin(other_pink_distances)
        pink_distance = other_pink_distances[nearest_index][0]
        
        other_white_distances = np.sqrt((other_white_nonzero[:,:,0] - tar_x) ** 2 + (other_white_nonzero[:,:,1] - tar_y) ** 2)
        nearest_index = np.argmin(other_white_distances)
        white_distance = other_white_distances[nearest_index][0]

        if white_distance > max_dist and pink_distance > max_dist:
            """ GROWTH (GREEN) OR DAMAGE (YELLOW) """
            canvas[tar_y, tar_x] = color1

        elif ispink==False and pink_distance < white_distance:
            """ BLEACHING (RED) """
            canvas[tar_y, tar_x] = color2
            
        elif ispink==True and white_distance < pink_distance:
            """ RECOVERY (BLUE) """
            canvas[tar_y, tar_x] = color2

        # cv2.imshow("canvas", canvas)
        # if cv2.waitKey(1) == 27:
        #     break

    return canvas

def get_diff(old_pink, old_white, new_pink, new_white, max_dist=30, close_ksize=0):
    oldpink_nonzero = get_nonzero(old_pink)
    oldwhite_nonzero = get_nonzero(old_white, to_open=True)
    newpink_nonzero = get_nonzero(new_pink)
    newwhite_nonzero = get_nonzero(new_white, to_open=True)

    height, width = new_pink.shape[:2]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # print("canvas.shape",canvas.shape)

    canvas = process_changes(canvas, newpink_nonzero , oldpink_nonzero, oldwhite_nonzero, ispink=True , color1=(0,255,0)  , color2=(255,0,0), max_dist=max_dist)
    canvas = process_changes(canvas, newwhite_nonzero, oldpink_nonzero, oldwhite_nonzero, ispink=False, color1=(0,255,0)  , color2=(0,0,255), max_dist=max_dist)
    canvas = process_changes(canvas, oldpink_nonzero , newpink_nonzero, newwhite_nonzero, ispink=True , color1=(0,255,255), color2=(0,0,255), max_dist=max_dist)
    canvas = process_changes(canvas, oldwhite_nonzero, newpink_nonzero, newwhite_nonzero, ispink=False, color1=(0,255,255), color2=(255,0,0), max_dist=max_dist)

    cv2.imshow("Canvas not opened", canvas)

    if close_ksize > 0:
        kernel = np.ones((close_ksize,close_ksize), np.uint8)
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("CanvasBGR opened", canvas)

    return canvas

def draw_diff(canvas, new_cropped, min_area=600):
    new_coral = new_cropped
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    contours, hier = cv2.findContours(canvas_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Canvas len(contours):", len(contours))

    for cont in contours:
        area = cv2.contourArea(cont)
        if area < min_area:
            continue
        print("Contour area:", area)

        x, y = cont[0][0]
        cont_b, cont_g, cont_r = canvas[y][x]
        cont_b, cont_g, cont_r =  int(cont_b), int(cont_g), int(cont_r)

        cont_x, cont_y, cont_w, cont_h = cv2.boundingRect(cont)

        cv2.rectangle(new_coral, (cont_x-10, cont_y-5), (cont_x+cont_w+5, cont_y+cont_h+20), (cont_b,cont_g,cont_r), 5)

        # cv2.imshow("new_coral", new_coral)
        # cv2.waitKey(0)

    return new_coral









if __name__ == "__main__":
    start_time = datetime.now()

    old_src = cv2.imread("./res/ref_cor.jpg")
    new_src = cv2.imread("out/out1/new_src.JPG")
    if old_src is None or new_src is None:
        exit("ERROR: failed to read image")
    
    cv2.imshow("old_src", auto_resize(old_src))
    cv2.imshow("new_src", auto_resize(new_src))

    old_cropped, _ = align_base(old_src, approx_height_ratio=4)
    new_cropped, crop_tuple = align_base(new_src, approx_height_ratio=5)

    cv2.imshow("old_cropped", old_cropped)
    cv2.imshow("new_cropped", new_cropped)
    
    cv2.imwrite("out/old_cropped.jpg", old_cropped)
    cv2.imwrite("out/new_cropped.jpg", new_cropped)

    align_base_time = datetime.now()
    print("Align base time:", align_base_time - start_time)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("")
    


    adjusting = False
    px_x, px_y = 0, 0

    # old_pink, old_white = eyedrop(old_cropped)
    old_pink, old_white = cv2.imread("temp_eye/old_pink.jpg", cv2.IMREAD_UNCHANGED), cv2.imread("temp_eye/old_white.jpg", cv2.IMREAD_UNCHANGED)
    ow_h, ow_w = old_white.shape[:2]
    cv2.rectangle(old_white, (0, ow_h//2), (ow_w, ow_h), (0,0,0), -1)

    # new_pink, new_white = eyedrop(new_cropped)
    new_pink, new_white = cv2.imread("temp_eye/new_pink.jpg", cv2.IMREAD_UNCHANGED), cv2.imread("temp_eye/new_white.jpg", cv2.IMREAD_UNCHANGED)
    
    print("old_pink.shape", old_pink.shape)
    print("new_pink.shape", new_pink.shape)

    def adjust_height(src, target):
        oc_h, oc_w = src.shape[:2]
        nc_h, nc_w = target.shape[:2]

        dh = nc_h - oc_h
        print("Difference in height:", dh)

        if dh <= 0:
            return src

        target_copy = np.zeros(target.shape, np.uint8)
        print("target_copy.shape", target_copy.shape)
        print('src.shape', src.shape)

        target_copy[dh:dh+oc_h, 0:oc_w] = src
        cv2.imshow("target_copy", target_copy)
        cv2.waitKey(0)

        return target_copy

    old_pink = adjust_height(old_pink , new_pink)
    old_white = adjust_height(old_white, new_white)

    cv2.imshow("old_pink", old_pink)
    cv2.imshow("old_white", old_white)
    cv2.imshow("new_pink", new_pink)
    cv2.imshow("new_white", new_white)

    if cv2.waitKey(0) == ord('s'):
        print("saving eyedrop results")
        cv2.imwrite("temp_eye/old_pink.jpg" , old_pink)
        cv2.imwrite("temp_eye/old_white.jpg", old_white)
        cv2.imwrite("temp_eye/new_pink.jpg" , new_pink)
        cv2.imwrite("temp_eye/new_white.jpg", new_white)
    cv2.destroyAllWindows()
    eyedrop_time = datetime.now()
    print("\nEyedrop time:", eyedrop_time - align_base_time)
    print("")



    canvas = get_diff(old_pink, old_white, new_pink, new_white, max_dist=70, close_ksize=0)
    # cv2.rectangle(canvas, (260,80), (330,180), (0,0,0), -1)
    cv2.imshow("canvas", canvas)
    new_drawn = draw_diff(canvas, new_cropped, min_area=650)

    # lower_x, lower_y, upper_x, upper_y = crop_tuple
    # out_new = new_src.copy()
    # out_new[lower_y:upper_y, lower_x:upper_x] = new_drawn
    # cv2.imshow("out_new", auto_resize(out_new))

    # cv2.imshow("old_src", auto_resize(old_src))
    # cv2.imshow("new_src", auto_resize(new_src))

    cv2.imshow("old_src", auto_resize(old_src))
    cv2.imshow("new_drawn", auto_resize(new_drawn, target_width=650))

    getdiff_time = datetime.now()
    print("\nGetdiff time:", getdiff_time - eyedrop_time)
    key = cv2.waitKey(0)
    if key == ord('s'):
        print("Saving to out dir")
        cv2.imwrite("out/old_src.jpg", old_src)
        cv2.imwrite("out/new_src.jpg", new_src)
        cv2.imwrite("out/old_cropped.jpg", old_cropped)
        cv2.imwrite("out/new_cropped.jpg", new_cropped)
        cv2.imwrite("out/old_pink.jpg", old_pink)
        cv2.imwrite("out/old_white.jpg", old_white)
        cv2.imwrite("out/new_pink.jpg", new_pink)
        cv2.imwrite("out/new_white.jpg", new_white)
        cv2.imwrite("out/canvas.jpg", canvas)
        cv2.imwrite("out/new_drawn.jpg", new_drawn)
    cv2.destroyAllWindows()
    print("End")