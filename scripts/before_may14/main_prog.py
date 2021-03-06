import cv2
import numpy as np

def cb_nothing(x):
    pass

def scale_resizing(img, scale):
    print("Original Dimension: ", img.shape)

    width = int(img.shape[1] * (scale/100))
    height= int(img.shape[0] * (scale/100))
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print("Resized Dimension: ", resized.shape)
    return resized

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

def click_event(event, x, y, flags, img):
    # https://docs.opencv.org/master/db/d5b/tutorial_py_mouse_handling.html
    """ Left click on image in hsv window to inspect a pixel once
        Right click to inspect pixels as you move cursor """
    global adjusting, temp_mask

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

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
def closing(img, ksize=5):
    kernel = np.ones((ksize,ksize), np.uint8)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closed

""" MAIN FUNCTIONS """
def background_remover(hsv):
    global adjusting, temp_mask

    cv2.namedWindow("Pixel Preview in HSV")
    cv2.namedWindow("Trackbar_Window")
    cv2.createTrackbar("hue_track", "Trackbar_Window", 30, 180, cb_nothing)
    cv2.createTrackbar("sat_track", "Trackbar_Window", 50, 255, cb_nothing)
    cv2.createTrackbar("val_track", "Trackbar_Window", 50, 255, cb_nothing)

    cv2.imshow("hsv", hsv)

    """ Generate pink mask """
    print("CLICK ON PINK PIXEL TO GET MASK FOR PINK CORAL")
    cv2.setMouseCallback('hsv', click_event, hsv)
    cv2.waitKey(0)
    pink_mask = temp_mask
    # cv2.imshow("pink_mask", pink_mask)

    """ Generate white mask """
    print("CLICK ON WHITE PIXEL TO GET MASK FOR WHITE CORAL")
    cv2.setMouseCallback('hsv', click_event, hsv)
    cv2.waitKey(0)
    white_mask = temp_mask
    # cv2.imshow("white_mask", white_mask)

    pink_white_mask = cv2.bitwise_or(pink_mask, white_mask)
    
    # cv2.imshow("combined_mask", pink_white_mask)

    return pink_white_mask

def alignment(gray):
    ret, thresh_gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    gray = closing(thresh_gray, ksize=5)

    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))

    if len(contours)==0:
        print("No contours found !!!")
        raise SystemExit()

    biggest_contour = contours[0]
    max_area = cv2.contourArea(biggest_contour)
    for cont in contours:
        this_area = cv2.contourArea(cont)
        if this_area > max_area:
            biggest_contour = cont
            max_area = this_area

    x, y, w, h = cv2.boundingRect(biggest_contour)
    # cv2.imshow("WIP_GRAY", gray)
    # cv2.drawContours(gray, [biggest_contour], -1, 0, -1)
    # cv2.imshow("DRAW_GRAY", gray)

    # TODO: hardcoding of dimension extension
    # To give some extension on window
    EXTEND_DIMENSION = 0.05
    x0 = int(x - (EXTEND_DIMENSION * w))
    y0 = int(y - (EXTEND_DIMENSION * h))
    x1 = int(x + ((1+EXTEND_DIMENSION) * w))
    y1 = int(y + ((1+EXTEND_DIMENSION) * h))

    print(x,y,x+w,y+h)
    print(x0,y0,x1,y1)

    contour_coordinates = np.float32([[x0,y0], [x1,y0], [x0,y1], [x1,y1]])

    height, width = h, w
    coordinates_new_img = np.float32([[0,0], [width,0], [0,height], [width,height]])

    matrix_perspective = cv2.getPerspectiveTransform(contour_coordinates, coordinates_new_img)
    perspective = cv2.warpPerspective(gray, matrix_perspective, (width, height))
    cv2.imshow("orig", gray)
    cv2.imshow("pers", perspective)

    # TODO: hardcoding target dimension
    TARGET_DIM = 600

    big_pers = scale_resizing(perspective, TARGET_DIM/perspective.shape[0]*100)
    # cv2.imshow("big_pers", big_pers)
    ret, thresh = cv2.threshold(big_pers, 127, 255, cv2.THRESH_BINARY)
    closed = closing(thresh, ksize=9)
    print(closed.shape)
    cv2.imshow("closed_Pers_transformation", closed)

    return closed

if __name__ == "__main__":
    IMAGES = ["coral_past.jpg", "black_box.jpg", "coral_dmg_mid.jpg", "front_flip.jpg", "coral_underwater.jpg"]

    """ Step 0: Reading Image Inputs """
    old = cv2.imread(IMAGES[0])
    new = cv2.imread(IMAGES[2])

    cv2.imshow("old", old)
    cv2.imshow("new", new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Images successfully read\n")

    old_hsv = cv2.cvtColor(old, cv2.COLOR_BGR2HSV)
    new_hsv = cv2.cvtColor(new, cv2.COLOR_BGR2HSV)

    """ Step 1: Background Removal 
        a. eyedropper function
        b. create mask based on eyedropper """

    # Initializing global variables to be used in mouse callback: pink and white mask
    adjusting = False # true if right clicked
    temp_mask = np.zeros((2,2), dtype=np.uint8)

    old_mask = background_remover(old_hsv)
    cv2.imshow("Old Mask in Main...", old_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    new_mask = background_remover(new_hsv)
    cv2.imshow("New Mask in Main...", new_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """ Step 2: Alignment of the two images 
        a. further isolate coral structure by identifying the biggest contour
        b. boundingRect on coral from contour
        c. use the (bottom) black structure for basis for alignment """

    # cv2.imwrite("OldMask.jpg", old_mask)
    # cv2.imwrite("NewMask.jpg", new_mask)

    old_mask_aligned = alignment(old_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    new_mask_aligned = alignment(new_mask)

    """ Step 3: Identify changes
        a. for growth and death: can use bitwise xor
        b. for bleaching and recovery: may need to use mean color? """

    cv2.waitKey(0)
    cv2.destroyAllWindows()