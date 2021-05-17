import cv2
import numpy as np
from numpy.lib.type_check import _imag_dispatcher

img_drawrect = cv2.imread("edited_src.jpg")

image_gray = cv2.cvtColor(img_drawrect, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(image_gray,127, 255,0)
#cv2.imshow("thresh", threshold)
# cv2.waitKey(0)

contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

# biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
# print(biggest_contour)
# cv2.drawContours(img_drawrect, biggest_contour, -1, [255,0,0], thickness=3)

# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(biggest_contour)
# w, h = img_drawrect.shape[::-1]
# top_left = min_loc
# bottom_right = (top_left[0] + w, top_left[1] + h)
# cv2.rectangle(img_drawrect,top_left, bottom_right, 255, 2)


# draw in blue the contours that were founded
cv2.drawContours(img_drawrect, contours, -1, 255, thickness=1)

# find the biggest countour (c) by the area
c = max(contours, key = cv2.contourArea)
x,y,w,h = cv2.boundingRect(c)

# draw the biggest contour (c) in green
cv2.rectangle(img_drawrect,(x-20,y-35),(x+w+20,y+h+20),(0,255,0),2)

cv2.imshow("rect", img_drawrect)
# cv2.waitKey(0)

src_src = cv2.imread("coral_under3.JPG")
rect = (x-20, y-35, w + 40, h+40)

mask = cv2.imread("edited_mask.jpg", cv2.IMREAD_GRAYSCALE)

""" TODO: Trying to remove the need for the colored image by using mask instead, but the threshold mask is empty """
# cv2.imshow("edited_mask", mask)

# minthresh = 1
# _, thresh_mask = cv2.threshold(mask, minthresh, 255, cv2.THRESH_BINARY)
# cv2.imshow("thresh_mask", thresh_mask)
# cv2.waitKey(0)
# mask = np.zeros(src_src.shape[:2], dtype = np.uint8)

# for i in range(0, 600):
#     for j in range(0, 800):
#         if np.any(in_mask[i, j] == 3):
#             cv2.circle(mask, (j-1, i-1), 1, 3, -1)
#         else:
#             cv2.circle(mask, (j-1, i-1), 1, 2, -1)

try:
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)
    # if (self.rect_or_mask == 0):         # grabcut with rect
    itercount = 1
    cv2.grabCut(src_src, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
    cv2.grabCut(src_src, mask, rect, bgdmodel, fgdmodel, itercount, cv2.GC_INIT_WITH_MASK)
        # self.rect_or_mask = 1
    # elif (self.rect_or_mask == 1):       # grabcut with mask
    # cv2.grabCut(src_src, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
except:
    import traceback
    traceback.print_exc()


mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')
if mask2 is None:
    exit("ERROR: mask2 None")

# print("src", src_src.shape)
# print("mask2", mask2.shape)
# print(mask2[0,0])

# mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
_, mask2 = cv2.threshold(mask2, 1, 255, cv2.THRESH_BINARY)
#cv2.imshow("thresh", mask2)
print(mask2.shape)
# cv2.waitKey(0)

output = cv2.bitwise_and(src_src, src_src, mask=mask2)

cv2.imshow("OUTPUT", output)
cv2.imwrite("out_rect.jpg", output)
cv2.waitKey(0)