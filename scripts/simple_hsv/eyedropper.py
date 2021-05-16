import cv2
import numpy as np

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
    global adjusting, temp_mask

    img = param

    if event==cv2.EVENT_RBUTTONDOWN:
        adjusting = not adjusting

    elif event==cv2.EVENT_LBUTTONDOWN or (event==cv2.EVENT_MOUSEMOVE and adjusting==True):
        hsv_val = img[y,x]
        print("Actual HSV Values:", hsv_val)
        hue = int(hsv_val[0])
        sat = int(hsv_val[1])
        val = int(hsv_val[2])

        pixel_preview = np.zeros((150,150,3), dtype=np.uint8)
        cv2.rectangle(pixel_preview, (0,0), (150,150), [hue,sat,val], -1)
        cv2.imshow("Pixel Preview in HSV", pixel_preview)

        # coor_string = str(x) + ',' + str(y)
        # hsv_val_string = "[{}, {}, {}]".format(hsv_val[0],hsv_val[1],hsv_val[2])
        # cv2.putText(img, hsv_val_string, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)

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
        print(lower, upper, '\n')

        temp_mask = cv2.inRange(img,lower,upper)
        cv2.imshow("temp_mask",temp_mask)

        cv2.imshow("hsv", img)

IMAGES = ["coral_past.jpg", "black_box.jpg", "front1.jpeg", "front_flip.jpg", "coral_underwater.jpg", "coral_dmg_left.jpg"]

if __name__ == "__main__":
    """ Still struggling finding good mask for underwater image i.e. IMAGES[4] """

    src = cv2.imread("../res/coral_under1.jpg")
    # src = cv2.resize(src, ( int(src.shape[1]*0.45), int(src.shape[0]*0.45) ), interpolation=cv2.INTER_AREA)
    cv2.imshow("src", src)

    # https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv
    # H: 0-179, S: 0-255, V: 0-255
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv", hsv)

    cv2.namedWindow("Pixel Preview in HSV")
    # cv2.resizeWindow("Pixel Preview in HSV", 300, 300)

    cv2.namedWindow("Trackbar_Window")
    cv2.createTrackbar("hue_track", "Trackbar_Window", 30, 180, cb_nothing)
    cv2.createTrackbar("sat_track", "Trackbar_Window", 50, 255, cb_nothing)
    cv2.createTrackbar("val_track", "Trackbar_Window", 50, 255, cb_nothing)

    """ Initializing global variables to be used in pink and white mask generation """
    adjusting = False # true if right clicked
    temp_mask = np.zeros((2,2), dtype=np.uint8)

    """ Generate pink mask """
    cv2.setMouseCallback('hsv', click_event, hsv)
    cv2.waitKey(0)
    pink_mask = temp_mask
    cv2.imshow("pink_mask", pink_mask)
    cv2.destroyWindow("temp_mask")

    """ Generate white mask """
    cv2.setMouseCallback('hsv', click_event, hsv)
    cv2.waitKey(0)
    white_mask = temp_mask
    cv2.imshow("white_mask", white_mask)
    cv2.destroyWindow("temp_mask")

    pw_mask = cv2.bitwise_or(pink_mask, white_mask)
    cv2.imshow("combined_mask", pw_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()