import cv2
import numpy as np

def scale_resizing(img, scale_percent):
    print("Original Dimension: ", img.shape)

    width = int(img.shape[1] * (scale_percent/100))
    height= int(img.shape[0] * (scale_percent/100))
    dim = (width, height)
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

    cv2.imshow("hsv", hsv)
    cv2.waitKey(0)

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
    cv2.imshow("base_mask", base_mask)

    return base_mask


def get_mid_line(src, mask):
    # gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), dtype=np.uint8))
    cv2.imshow("closed", closed)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("no. of contours: ", len(contours))

    # find the biggest countour (c) by area
    c = max(contours, key = cv2.contourArea)

    # # extLeft = tuple(c[c[:, :, 0].argmin()][0])
    # # extRight = tuple(c[c[:, :, 0].argmax()][0])
    # # extTop = tuple(c[c[:, :, 1].argmin()][0])
    # # extBot = tuple(c[c[:, :, 1].argmax()][0])

    # canvas = np.ones(base_mask.shape, dtype=np.uint8)
    # print("canvas.shape:", canvas.shape)
    # # https://stackoverflow.com/questions/45246036/cv2-drawcontours-will-not-draw-filled-contour
    # cv2.drawContours(canvas, [c], -1, [255,255,255], thickness=-1)
    # # cv2.circle(canvas, extLeft, 8, (0, 0, 255), -1)     # red
    # # cv2.circle(canvas, extRight, 8, (0, 255, 0), -1)    # green
    # # cv2.circle(canvas, extTop, 8, (255, 0, 0), -1)      # blue
    # # cv2.circle(canvas, extBot, 8, (255, 255, 0), -1)    # cyan

    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(src, (x,y), (x+w,y+h), (255,0,0), 2)

    height, width = src.shape[:2]
    cv2.line(src, (x+w//2, 0), (x+w//2, height), (0,0,255), 2)

    # cv2.namedWindow("canvas", cv2.WINDOW_NORMAL)
    # cv2.imshow("canvas", canvas)

    cv2.imshow("line", src)
    cv2.waitKey(0)


src = cv2.imread("new_coral1.JPG")
if src is None:
    exit("ERROR: failed to read image")

src = scale_resizing(src, scale_percent=20)

hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

base_mask = get_black_base(hsv)
print("base_mask.shape:", base_mask.shape)
# black_base_with_noise = cv2.bitwise_and(src, src, mask=base_mask)

get_mid_line(src, base_mask)