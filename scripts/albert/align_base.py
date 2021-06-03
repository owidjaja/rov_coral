import cv2
import numpy as np

def auto_resize(img, target_width=800):
    print("Original Dimension: ", img.shape)

    orig_height, orig_width = img.shape[:2]
    scale_ratio = target_width / orig_width

    new_width = int(img.shape[1] * (scale_ratio))
    new_height= int(img.shape[0] * (scale_ratio))
    dim = (new_width, new_height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print("Resized Dimension: ", resized.shape, '\n')
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
    print("actual:", actual_hsv)

    hue_tol, sat_tol, val_tol = tolerance
    print("tolerance:", tolerance)

    hue_upper = extend_range(hue, 1, 'u', hue_tol)
    hue_lower = extend_range(hue, 1, 'l', hue_tol)
    sat_upper = extend_range(sat, 0, 'u', sat_tol)
    sat_lower = extend_range(sat, 0, 'l', sat_tol)
    val_upper = extend_range(val, 0, 'u', val_tol)
    val_lower = extend_range(val, 0, 'l', val_tol)

    upper =  np.array([hue_upper, sat_upper, val_upper])
    lower =  np.array([hue_lower, sat_lower, val_lower])
    print("range:", lower, upper, '\n')

    base_mask = cv2.inRange(hsv, lower, upper)
    # cv2.imshow("base_mask", base_mask)
    
    cv2.waitKey(0)

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
        print("no. of contours found: ", num_contours)

    # find the biggest countour (c) by area
    c = max(contours, key = cv2.contourArea)

    x,y,w,h = cv2.boundingRect(c)
    # cv2.rectangle(src, (x,y), (x+w,y+h), (255,0,0), 2)

    height, width = src.shape[:2]
    # cv2.line(src, (x+w//2, 0), (x+w//2, height), (0,0,255), 2)

    # cv2.imshow("line", src)
    # cv2.waitKey(0)

    return [x, y, w, h]

def crop_to_standard(src, base_dim, approx_height_ratio=4, crop_extend=0.1):
    x, y, w, h = base_dim

    lower_x = int(x - crop_extend * w)
    lower_y = int(y - (approx_height_ratio + crop_extend) * h)

    upper_x = int(x + (1 + crop_extend) * w)
    upper_y = int(y + (1 + crop_extend) * h)

    cropped = src[lower_y:upper_y, lower_x:upper_x]
    cropped = auto_resize(cropped, target_width=360)

    cv2.imshow("cropped", cropped)
    k = cv2.waitKey(0)
    if k == ord('s'):
        print("saving cropped image with dim:", cropped.shape)
        cv2.imwrite("cropped.jpg", cropped)

src = cv2.imread("new_coral1.JPG")
if src is None:
    exit("ERROR: failed to read image")

# src = auto_resize(src)

hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

base_mask = get_black_base(hsv)

base_dim = get_mid_line(src, base_mask)

crop_to_standard(src, base_dim)