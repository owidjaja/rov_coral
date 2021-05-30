import cv2
import numpy as np

""" Custom grabcut with custom masks """

img_drawrect = cv2.imread("edited_src.jpg")

image_gray = cv2.cvtColor(img_drawrect, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(image_gray, 127, 255,0)
#cv2.imshow("thresh", threshold)
# cv2.waitKey(0)

ksize = 5
my_kernel = np.ones((ksize,ksize), np.uint8)
closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, my_kernel)

contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

# draw in blue the contours that were found
# cv2.drawContours(img_drawrect, contours, -1, [255,0,0], thickness=2)

# find the biggest countour (c) by area
c = max(contours, key = cv2.contourArea)
print(len(c))
cv2.drawContours(img_drawrect, c, -1, [255,0,255], thickness=3)
x,y,w,h = cv2.boundingRect(c)

# draw green rect around the biggest contour (c)
# need to extend out ceiling as as white coral poorly detected
extend_range = 10
cv2.rectangle(img_drawrect,(x-extend_range,y-extend_range),(x+w+extend_range,y+h+extend_range),(0,255,0),2)

cv2.imshow("rect", img_drawrect)
# cv2.waitKey(0)

src_src = cv2.imread("coral_under3.JPG")
rect = (x-extend_range, y-extend_range, w+2*extend_range, h+2*extend_range)

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

rect_or_mask = 'r'
output = np.zeros((src_src.shape), dtype=np.uint8)
while True:
    cv2.imshow('output', output)
    cv2.imshow('input', src_src)
    k = cv2.waitKey(1)
    if k == 27:        # esc to exit
        break
    elif k == ord('n'):
        try:
            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)
            if (rect_or_mask == 'r'):         # grabcut with rect in first loop iter
                cv2.grabCut(src_src, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
                rect_or_mask = 'm'
            else:
                cv2.grabCut(src_src, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
        except:
            import traceback
            traceback.print_exc()

    mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')
    if mask2 is None:
        exit("ERROR: mask2 None")
    output = cv2.bitwise_and(src_src, src_src, mask=mask2)

cv2.imshow("OUTPUT", output)
if cv2.waitKey(0) == ord('s'):
    print("saving output as out_rect.jpg...")
    cv2.imwrite("out_rect.jpg", output)
