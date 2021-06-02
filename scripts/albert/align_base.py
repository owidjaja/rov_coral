import cv2
import numpy as np

base_mask = cv2.imread("black_base.jpg")
print("base_mask.shape", base_mask.shape)
cv2.namedWindow("base_mask", cv2.WINDOW_NORMAL)
cv2.imshow("base_mask", base_mask)


gray = cv2.cvtColor(base_mask, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3,3), dtype=np.uint8))

contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("no. of contours: ", len(contours))

# find the biggest countour (c) by area
c = max(contours, key = cv2.contourArea)

# extLeft = tuple(c[c[:, :, 0].argmin()][0])
# extRight = tuple(c[c[:, :, 0].argmax()][0])
# extTop = tuple(c[c[:, :, 1].argmin()][0])
# extBot = tuple(c[c[:, :, 1].argmax()][0])

canvas = np.ones(base_mask.shape, dtype=np.uint8)
print("canvas.shape", canvas.shape)
# https://stackoverflow.com/questions/45246036/cv2-drawcontours-will-not-draw-filled-contour
cv2.drawContours(canvas, [c], -1, [255,255,255], thickness=-1)
# cv2.circle(canvas, extLeft, 8, (0, 0, 255), -1)     # red
# cv2.circle(canvas, extRight, 8, (0, 255, 0), -1)    # green
# cv2.circle(canvas, extTop, 8, (255, 0, 0), -1)      # blue
# cv2.circle(canvas, extBot, 8, (255, 255, 0), -1)    # cyan

x,y,w,h = cv2.boundingRect(c)
cv2.rectangle(base_mask, (x,y), (x+w,y+h), (255,0,0), 2)

height, width = canvas.shape[:2]
cv2.line(canvas, (x+w//2, 0), (x+w//2, height), (0,0,255), 2)

cv2.namedWindow("canvas", cv2.WINDOW_NORMAL)
cv2.imshow("canvas", canvas)



cv2.waitKey(0)