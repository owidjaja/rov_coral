import cv2
import numpy as np
from matplotlib import pyplot as plt

def boundingRect_coral(img):
    """ Produce boundingRect for just the coral structure """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    my_mask = cv2.inRange(hsv, np.array([0,30,136]), np.array([255,255,255]))
    my_mask = cv2.inRange(hsv, np.array([0,0,203]), np.array([190,96,255]))
    # cv2.imshow("mask", my_mask)

    res = cv2.bitwise_and(img, img, mask=my_mask)
    # cv2.imshow("res", res)

    # https://stackoverflow.com/questions/41879315/opencv-visualize-polygonal-curves-extracted-with-cv2-approxpolydp
    thresh = my_mask

    canvas = np.zeros(img.shape, np.uint8)
    img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # find biggest contour based on area
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))
    # print(hierarchy)
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)

    x, y, w, h = cv2.boundingRect(cnt)

    ERROR_MARGIN = 25
    x -= ERROR_MARGIN
    y -= ERROR_MARGIN
    w += ERROR_MARGIN
    h += ERROR_MARGIN

    cv2.rectangle(img, (x,y), (x+w,y+h), [0,255,0])

    return x, y, w, h


def perspective_transformation(img, x, y, w, h):
    # https://www.youtube.com/watch?v=mzhiKpM8eJ0
    # https://www.youtube.com/watch?v=j4el1XARYSo

    rows, cols, ch = img.shape

    coor_orig_img = np.float32([[x,y], [x+w,y], [x,y+h], [x+w,y+h]])
    for x in range(4):
        cv2.circle(img, (int(coor_orig_img[x][0]), int(coor_orig_img[x][1])), 5, (255,0,0), -1)

    # cv2.imshow("Input", img)
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("Input")

    height, width = h, w
    coor_new_img = np.float32([[0,0], [width,0], [0,height], [width,height]])

    # get perspective transformation matrix
    matrix_perspective = cv2.getPerspectiveTransform(coor_orig_img, coor_new_img)   

    # perform transformation
    perspective = cv2.warpPerspective(img, matrix_perspective, (width, height))

    # cv2.imshow("Output", perspective)
    plt.subplot(122), plt.imshow(cv2.cvtColor(perspective, cv2.COLOR_BGR2RGB)), plt.title("Output")


if __name__ == "__main__":
    IMAGES = ["coral_past.jpg", "front_flip.jpg", "getperspective_transform_01.jpg"]
    img = cv2.imread(IMAGES[1])

    x, y, w, h = boundingRect_coral(img)
    perspective_transformation(img, x, y, w, h)

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()