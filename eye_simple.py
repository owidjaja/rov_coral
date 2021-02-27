import cv2
import numpy as np

def click_event(event, x, y, flags, param):
    # https://docs.opencv.org/master/db/d5b/tutorial_py_mouse_handling.html
    """ Left click on image in hsv window to inspect a pixel once
        Right click to inspect pixels as you move cursor """

    img = param

    if event==cv2.EVENT_LBUTTONDOWN:
        hsv_val = img[y,x]
        print("Color Values:", hsv_val)
        hue = int(hsv_val[0])
        sat = int(hsv_val[1])
        val = int(hsv_val[2])

        pixel_preview = np.zeros((150,150,3), dtype=np.uint8)
        cv2.rectangle(pixel_preview, (0,0), (150,150), [hue,sat,val], -1)
        cv2.imshow("Pixel Preview in HSV", pixel_preview)

        cv2.imshow("hsv", img)

        TOLERANCE = 10
        lower =  np.array([hue-TOLERANCE, 0, 0])
        upper =  np.array([hue+TOLERANCE, 255, 255])

        temp_mask = cv2.inRange(img,lower,upper)
        cv2.imshow("temp_mask",temp_mask)

IMAGES = ["coral_past.jpg", "black_box.jpg", "front1.jpeg", "front_flip.jpg", "coral_underwater.jpg", "coral_dmg_left.jpg", "star.jpeg"]

if __name__ == "__main__":
    """ Still struggling finding good mask for underwater image i.e. IMAGES[4] """

    src = cv2.imread(IMAGES[6])

    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv", hsv)

    cv2.setMouseCallback('hsv', click_event, hsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()