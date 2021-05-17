import cv2
import numpy as np

def hsv_extend_range(value, is_type_hue, upper_or_lower, tolerance):
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

def extend_range(value, upper_or_lower, tolerance, boundary=255):

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

img = cv2.imread('../res/coral_under3.jpg',cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("src", img)

data = np.reshape(img, (-1,3))
print(data.shape)
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 4, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness,labels,centers = cv2.kmeans(data,1,None,criteria,4,flags)

dominant_ = centers[0]
print('Dominant color is: bgr({})'.format(dominant_.astype(np.int32)))

tolerance = 20
# b_upper = round(extend_range(dominant_bgr[0], 'u', tolerance))
# b_lower = round(extend_range(dominant_bgr[0], 'l', tolerance))
# g_upper = round(extend_range(dominant_bgr[1], 'u', tolerance))
# g_lower = round(extend_range(dominant_bgr[1], 'l', tolerance))
# r_upper = round(extend_range(dominant_bgr[2], 'u', tolerance))
# r_lower = round(extend_range(dominant_bgr[2], 'l', tolerance))

# upper =  np.array([b_upper, g_upper, r_upper])
# lower =  np.array([b_lower, g_lower, r_lower])

hue = dominant_[0]
sat = dominant_[1]
val = dominant_[2]

hue_upper = extend_range(hue, 1, 'u', HUE_TOLERANCE)
hue_lower = extend_range(hue, 1, 'l', HUE_TOLERANCE)
sat_upper = extend_range(sat, 0, 'u', SAT_TOLERANCE)
sat_lower = extend_range(sat, 0, 'l', SAT_TOLERANCE)
val_upper = extend_range(val, 0, 'u', VAL_TOLERANCE)
val_lower = extend_range(val, 0, 'l', VAL_TOLERANCE)

upper =  np.array([hue_upper, sat_upper, val_upper])
lower =  np.array([hue_lower, sat_lower, val_lower])
print(lower, upper, '\n')

dom_mask = cv2.inRange(img, lower, upper)
cv2.imshow("dom_mask", dom_mask)

dom_res = cv2.bitwise_and(img, img, mask=dom_mask)
cv2.imshow("dom_res", dom_res)

cv2.waitKey(0)