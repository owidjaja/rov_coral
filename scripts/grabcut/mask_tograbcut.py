import cv2
import numpy as np
from matplotlib import pyplot as plt

""" take in input mask: coral_mask.jpg for rect, background_mask.jpg for mask
    and edit values onto mask to simulate grabcut rect mask """

def get_rect():
    """
    take in mask from hsv to get approximate rect around coral
    replaces the edited_src.jpg, substitute for yolo
    """

    # read as 3 channel as rect later on drawn in purple
    img_drawrect = cv2.imread("coral_mask.jpg")
    if img_drawrect is None:
        exit("ERROR: failed to read img_drawrect")

    # need gray for contour operation
    image_gray = cv2.cvtColor(img_drawrect, cv2.COLOR_BGR2GRAY)

    # closed = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    closed = image_gray

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # draw in blue the contours that were found
    # cv2.drawContours(img_drawrect, contours, -1, [255,0,0], thickness=2)

    # find the biggest countour (c) by area
    c = max(contours, key = cv2.contourArea)
    cv2.drawContours(img_drawrect, c, -1, [255,0,255], thickness=3)
    x,y,w,h = cv2.boundingRect(c)

    # need to extend out ceiling as as white coral poorly detected
    # TODO: make function for extend range to handle edge cases of <0 and >width, >height
    extend_range = 10
    ext_x = x-extend_range
    ext_y = y-extend_range
    ext_w = w+2*extend_range
    ext_h = h+2*extend_range

    rect = (ext_x, ext_y, ext_w, ext_h)
    print ("rect", rect)

    # draw green rect around the biggest contour (c)
    # cv2.rectangle(img_drawrect,(x-extend_range,y-extend_range),(x+w+extend_range,y+h+extend_range),(0,255,0),2)
    cv2.rectangle(img_drawrect, (ext_x, ext_y), (ext_x+ext_w, ext_y+ext_h), (0,255,0), thickness=2)

    # cv2.imshow("rect", img_drawrect)
    # cv2.waitKey(0)

    return rect

def get_mask(rect):
    """ convert background_mask.jpg to appropriate mask for grabcut """

    background_mask = cv2.imread("background_mask.jpg", 0)

    # to clean up stray pixels
    _, background_mask = cv2.threshold(background_mask, 127, 255, cv2.THRESH_BINARY)

    print("background_mask.shape", background_mask.shape)
    height, width = background_mask.shape[:2]

    # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
    white_pixels = np.where(background_mask[:, :] != 0)

    # out_mask initialized as probable foreground
    out_mask = np.zeros(background_mask.shape, dtype=np.uint8)
    out_mask[:,:] = 3
    """
        0: background 
        1: foreground
        2: probable background
        3: probable foreground
    """
    out_mask[white_pixels] = 2      # probable background based on blue background mask

    # drawing definite background (0) on out_mask, all pixels outside rect
    img_drawrect = np.zeros(background_mask.shape, dtype=np.uint8)
    cv2.rectangle(img_drawrect, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), 255, thickness=-1)
    out_mask[np.where(img_drawrect==0)] = 0
    # cv2.imshow("img_drawrect in get mask", img_drawrect)

    combined_background = np.where((out_mask==0) + (out_mask==2), 255, 0).astype('uint8')
    combined_foreground = cv2.bitwise_not(combined_background)

    # cv2.imshow("background_mask", background_mask)
    # cv2.imshow("actual out_mask", out_mask)
    # cv2.imshow("definite_background", np.where(out_mask==0, 255, 0).astype('uint8'))
    # cv2.imshow("probable_background", np.where(out_mask==2, 255, 0).astype('uint8'))
    # cv2.imshow("combined_foreground", combined_foreground)

    return out_mask


img_src = cv2.imread("coral_under3.jpg")
rect = get_rect()
mask = get_mask(rect)

rect_or_mask = 'r'
output = np.zeros((img_src.shape), dtype=np.uint8)
for i in range(3):
# while True:
    cv2.imshow('output', output)
    cv2.imshow('input', img_src)
    k = cv2.waitKey(1)

    if k == 27:        # esc to exit
        break
    # elif k == ord('n'):
    try:
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)
        cv2.grabCut(img_src, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
    except:
        import traceback
        traceback.print_exc()

    mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')
    # cv2.imshow("mask2 = probable + definite foreground", mask2)
    cv2.imshow("background", np.where(mask==0, 255, 0).astype('uint8'))
    cv2.imshow("foreground", np.where(mask==1, 255, 0).astype('uint8'))
    cv2.imshow("probable_background", np.where(mask==2, 255, 0).astype('uint8'))
    cv2.imshow("probable_foreground", np.where(mask==3, 255, 0).astype('uint8'))
    output = cv2.bitwise_and(img_src, img_src, mask=mask2)

if cv2.waitKey(0) == ord('s'):
    pass