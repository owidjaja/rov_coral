import cv2
import numpy as np

def extend_range(value, is_type_hue, upper_or_lower):
    # https://github.com/alieldinayman/HSV-Color-Picker/blob/master/HSV%20Color%20Picker.py
    # https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv

    """ Assign ranges based on hsv type and whether the range is upper or lower, with hardcoded tolerance value
        Still having trouble finding appropriate tolerance value for real underwater image """

    # TODO: Need to adjust tolerance value, may consider using trackbar 
    TOLERANCE = 25

    if is_type_hue == 1:
        # set the boundary for hue (0-180)°
        boundary = 180
    else:
        # set the boundary for saturation and value (0-255)
        boundary = 255

    if upper_or_lower == 'u':
        value = value + TOLERANCE
    elif upper_or_lower == 'l':
        value = value - TOLERANCE

    # adjust for extreme values
    if(value + TOLERANCE > boundary):
        value = boundary
    elif (value - TOLERANCE < 0):
        value = 0

    return value

def click_event(event, x, y, flags, img):
    """ Click on image in hsv window """

    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_val = img[y,x]
        print("Actual HSV Values:", hsv_val)

        coor_string = str(x) + ',' + str(y)
        hsv_val_string = "[{}, {}, {}]".format(hsv_val[0],hsv_val[1],hsv_val[2])

        # cv2.putText(img, hsv_val_string, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
        
        hue_upper = extend_range(hsv_val[0], 1, 'u')
        hue_lower = extend_range(hsv_val[0], 1, 'l')
        sat_upper = extend_range(hsv_val[1], 0, 'u')
        sat_lower = extend_range(hsv_val[1], 0, 'l')
        val_upper = extend_range(hsv_val[2], 0, 'u')
        val_lower = extend_range(hsv_val[2], 0, 'l')

        upper =  np.array([hue_upper, sat_upper, val_upper])
        lower =  np.array([hue_lower, sat_lower, val_lower])
        print(lower, upper, '\n')

        image_mask = cv2.inRange(img,lower,upper)
        cv2.imshow("Mask",image_mask)

        cv2.imshow("hsv", img)

IMAGES = ["coral_past.jpg", "black_box.jpg", "front1.jpeg", "front_flip.jpg", "coral_underwater.jpg"]

def main():
    """ Still struggling finding good mask for underwater image i.e. IMAGES[4] """

    src = cv2.imread(IMAGES[0])
    # cv2.imshow("src", src)

    # https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv
    # H: 0-179, S: 0-255, V: 0-255
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv", hsv)

    cv2.setMouseCallback('hsv', click_event, hsv)



    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()