import cv2
import numpy as np

COLOR_GREEN  = [0,255,0]    # GROWTH
COLOR_YELLOW = [0,255,255]  # DAMAGE/DEATH
COLOR_RED    = [0,0,255]    # BLEACH
COLOR_BLUE   = [255,0,0]    # RECOVERY

def background_remover(img):
    """ Hardcode hsv values to generate mask for only the coral structure (white and pink) """

    # TODO: hardcoding hsv values
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    my_mask = cv2.inRange(hsv, np.array([0,30,136]), np.array([255,255,255]))
    # cv2.imshow("mask", my_mask)
    res = cv2.bitwise_and(img, img, mask=my_mask)
    # cv2.imshow("res", res)

    return res

def remove_noise(img, ksize=5, thresh_val=32):
    """ Remove noise from img using medianBlur """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, ksize)
    # cv2.imshow("blurred", blur)
    _, thresh = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh)

    # https://stackoverflow.com/questions/42920810/in-opencv-is-mask-a-bitwise-and-operation
    res = cv2.bitwise_and(img, img, mask=thresh)
    return res

def closing(img, ksize=5):
    kernel = np.ones((ksize,ksize), np.uint8)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closed

def get_mean(img, cont):
    """ Get mean colour of pixels in contour passed in parameter """

    # TODO: find mean color: if pink then growth, if white then bleached
    # https://stackoverflow.com/questions/54316588/get-the-average-color-inside-a-contour-with-open-cv
    # https://stackoverflow.com/questions/15341538/numpy-opencv-2-how-do-i-crop-non-rectangular-region

    cont_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(cont_mask, [cont], -1, 255, -1)
    # cont_mask = cv2.cvtColor(cont_mask, cv2.COLOR_BGR2GRAY)
    # _, cont_mask = cv2.threshold(cont_mask, 32, 255, cv2.THRESH_BINARY)
    mean = cv2.mean(cv2.cvtColor(new_coral, cv2.COLOR_BGR2HSV), mask=cont_mask)
    print(mean)

    cv2.imshow("mean_mask", cv2.bitwise_and(new_coral, new_coral, mask=cont_mask))
    cv2.waitKey(0)
    return mean

def compare_mean(curr_coral, xor_diff_coral, cont):
    """ Compare contour mean between structure in xor_diff and in current coral
        Returns False if difference is below a certain value                    """
    
    x, y, w, h = cv2.boundingRect(cont)

    # cropping to contour as ROI
    cont_img = xor_diff_coral[y:y+h, x:x+w]
    
    # only need to grab mean()[0] because dealing with binary images
    cont_mean = cv2.mean(cont_img)[0]
    # print("cm", cont_mean)
    # cv2.imshow("temp1", cont_img)

    mask_for_contour_in_img = np.zeros(curr_coral.shape[:2], np.uint8)
    cv2.rectangle(mask_for_contour_in_img, (x,y), (x+w,y+h), 255, -1)
    # cv2.imshow("temp2", mask_for_contour_in_img)

    img_section_mean = cv2.mean(curr_coral, mask_for_contour_in_img)[0]
    # print("ism", img_section_mean)
    cv2.waitKey(0)

    # TODO: hardcoding of difference value
    if abs(cont_mean-img_section_mean) < 5:
        return False
    else:
        return True


IMAGES = ["coral_past.jpg", "black_box.jpg", "front1.jpeg", "front_flip.jpg", "coral_underwater.jpg"]

if __name__ == "__main__":
    
    """ Choose which two images to compare from here
        Working ones for now are IMAGES[0] and IMAGES[1]
        Need to work on resizing to a standard size for the rest """

    old = cv2.imread("../albert/eyedrop_pink_mask.jpg")
    new = cv2.imread("../albert/new_pink_mask.jpg")
    cv2.imshow("old", old)
    cv2.imshow("new", new)
    cv2.waitKey(0)

    old_coral = background_remover(old)
    new_coral = background_remover(new)
    cv2.imshow("old_cor", old_coral)
    cv2.imshow("new_cor", new_coral)
    cv2.waitKey(0)

    # TODO: resizing images so that these old_coral, new_coral would always be same size
    # diff = cv2.bitwise_xor(old_coral, new_coral)
    diff = cv2.bitwise_xor(old, new)
    cv2.imshow("diff", diff)

    ref_diff = remove_noise(diff)

    # temporary closing operation to glob structures together (avoid seperate contours)
    # https://stackoverflow.com/questions/55107660/python-cv2-find-contours-minimum-dimensions
    closed_diff = closing(ref_diff)
    # cv2.imshow("closed", closed_diff)

    contours, _ = cv2.findContours(cv2.cvtColor(closed_diff, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(ref_diff, contours, -1, (0, 255, 0), 1)
    # print(len(contours))
    # cv2.imshow("refined", ref_diff)

    new_coral_gray = cv2.cvtColor(new_coral, cv2.COLOR_BGR2GRAY)
    _, new_coral_bin = cv2.threshold(new_coral_gray, 32, 255, cv2.THRESH_BINARY)

    # detecting whether structure in xor_diff is present in current_coral
    # drawing corresponding rectangles
    for cont in contours:
        # TODO: hardcoding of minimum size of contour to be considered
        area = cv2.contourArea(cont)
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(cont)

        """ Method 1: finding difference of mean """
        if compare_mean(new_coral_bin, closed_diff, cont) == False:
            # new structure: growth
            cv2.rectangle(new, (x,y), (x+w,y+h), COLOR_GREEN, 5)
        else:
            # contour absent in current coral: death
            cv2.rectangle(new, (x,y), (x+w,y+h), COLOR_YELLOW, 5)

        """ Method 2: differentiating using mean of contour color """
        # mean = get_mean(new_coral_bin[x:y, x+w:y+h], cont)

        # mean = get_mean(new_coral, cont)
        # hue = mean[0]
        # sat = mean[1]
        # val = mean[2]
        
        # if sat<50 and val>200:
        #     # if pipe is white: bleached
        #     cv2.rectangle(new, (x,y), (x+w,y+h), COLOR_RED, 5)
        # elif 140<hue and hue<180:
        #     # if hue is pink (color of pipe): growth
        #     cv2.rectangle(new, (x,y), (x+w,y+h), COLOR_GREEN, 5)
        # else:
        #     # else death
        #     cv2.rectangle(new, (x,y), (x+w,y+h), COLOR_YELLOW, 5)
        #     # cv2.rectangle(old, (x,y), (x+w,y+h), [0,255,0], 5)



        # TODO: mask pink corals [150,60,90], [255,255,255]
        

    cv2.imshow("old", old)
    cv2.imshow("new", new)

    cv2.waitKey(0)
    cv2.destroyAllWindows()