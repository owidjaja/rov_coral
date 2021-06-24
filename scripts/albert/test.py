import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
from numpy.core.arrayprint import ComplexFloatingFormat

""" revisiting bitwise... """
# pink  = cv2.imread("out/new_pink.jpg")
# white = cv2.imread("out/old_white.jpg")

# height, width = white.shape[:2]
# pink = cv2.resize(pink, (width, height))

# print(pink.shape)
# print(white.shape)

# cv2.imshow("pink", pink)
# cv2.imshow("white", white)

# comb = cv2.bitwise_or(pink, white)
# cv2.imshow("comb", comb)

# canvas = cv2.imread("out/canvas.jpg")
# cv2.imshow("canvas", canvas)

# ksize = 5
# kernel = np.ones((ksize,ksize), dtype=np.uint8)
# closed = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
# cv2.imshow("closed", closed)

""" placing on top of frame """
# old = cv2.imread("out/old_pink.jpg")
# cv2.imshow("old", old)
# new = cv2.imread("out/new_pink.jpg")
# cv2.imshow("new", new)

# oh, ow = old.shape[:2]
# nh, nw = new.shape[:2]

# dh = nh - oh
# print(dh)

# new_copy = np.zeros(new.shape, np.uint8)
# cv2.imshow("nc", new_copy)

# new_copy[dh:dh+oh, 0:ow] = old
# cv2.imshow("nc_edit", new_copy)


""" outdated draw contour script for combined_prog.py """
    # for cont in contours:
    #     area = cv2.contourArea(cont)
    #     if area < min_area:
    #         continue
    #     print("Contour area:", area)

    #     x, y = cont[0][0]
    #     cont_b, cont_g, cont_r = canvas[y][x]
    #     cont_b, cont_g, cont_r =  int(cont_b), int(cont_g), int(cont_r)

    #     cont_x, cont_y, cont_w, cont_h = cv2.boundingRect(cont)

    #     cv2.rectangle(new_coral, (cont_x-10, cont_y-5), (cont_x+cont_w+5, cont_y+cont_h+20), (cont_b,cont_g,cont_r), 5)

    #     # cv2.imshow("new_coral", new_coral)
    #     # cv2.waitKey(0)


""" get top 4 contours 
    https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/ """

def draw_contour(image, c, i):
	# compute the center of the contour area and draw a circle
	# representing the center
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	# draw the countour number on the image
	cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (255, 255, 255), 2)
	# return the image with the contour number drawn on it
	return image

# img = cv2.imread("res/canvas.jpg")

# img_copy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, img_copy = cv2.threshold(img_copy, 32, 255, cv2.THRESH_BINARY)

# cnts = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# height, width = img.shape[:2]
# canvas = np.zeros((height,width,3), np.uint8)

# # contours = cnts[:4]
# contours = cnts
# for (i,c) in enumerate(contours):
#     # draw_contour(img, c, i)
#     pass

# min_area = 650
# for cont in contours:
#     area = cv2.contourArea(cont)
#     # if area < min_area:
#     #     continue
#     print("Contour area:", area)

#     x, y = cont[0][0]
#     # cv2.circle(img, (x,y), 10, (255,255,255), -1)
#     # cont_b, cont_g, cont_r = img[y][x]
#     # cont_b, cont_g, cont_r =  int(cont_b), int(cont_g), int(cont_r)
#     # print("bgr", cont_b, cont_g, cont_r)

#     cont_bgr = img[y][x]
#     max_bgr = max(cont_bgr)
#     print("cont_bgr", cont_bgr)
#     print("max_bgr", max_bgr)

#     cb, cg, cr = cont_bgr
#     cb, cg, cr = int(cb), int(cg), int(cr)  # convert np.uint8 into python int

#     print("abs(cg-cr):", abs(cg-cr))
#     if abs(cg - cr) <= 15:
#         color = (0,255,255)
#     elif cb == max_bgr:
#         color = (255,0,0)
#     elif cg == max_bgr:
#         color = (0,255,0)
#     else:
#         color = (0,0,255)

#     cont_x, cont_y, cont_w, cont_h = cv2.boundingRect(cont)

#     cv2.rectangle(img, (cont_x-10, cont_y-5), (cont_x+cont_w+5, cont_y+cont_h+20), color, 5)
#     print("")

# cv2.imshow("img Text", img)

""" nice canvas.jpg """
YELLOW = (0,255,255)
RED = (0,0,255)

canvas = cv2.imread("out/out1/canvas.jpg")
# cv2.rectangle(canvas, (20,90), (90,244), (255,0,0), -1)

roi = canvas[90:244, 20:90]

red = np.where(
    # (roi[:, :, 0] > 240) & 
    (roi[:, :, 1] < 240) &
    (roi[:, :, 2] > 30)
)

# print(red)
roi[red] = YELLOW

# cv2.circle(canvas, (102,247), 20, RED, -1)

cv2.imshow("canvas", canvas)
cv2.imshow("roi", roi)

plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
plt.show()

if cv2.waitKey(0) == ord('s'):
    cv2.imwrite("canvas.jpg", canvas)