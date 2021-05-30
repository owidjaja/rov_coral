import cv2
import numpy as np

""" result: background blue mask hard to produce
pink hsv actual: 104 75 123
white hsv actual: 94 125 155
"""

img = cv2.imread("../res/coral_under3.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv", hsv)

low_hue = cv2.inRange(hsv, (0,0,0), (75,255,255))
cv2.imshow("low_hue", low_hue)
high_hue = cv2.inRange(hsv, (125,0,0), (180,255,255))
cv2.imshow("high_hue", high_hue)

low_sat = cv2.inRange(hsv, (0,0,0), (180,30,255))
cv2.imshow("low_sat", low_sat)
high_sat = cv2.inRange(hsv, (0,150,0), (180,255,255))
cv2.imshow("high_sat", high_sat)

low_val = cv2.inRange(hsv, (0,0,0), (180,255,100))
cv2.imshow("low_val", low_val)
high_val = cv2.inRange(hsv, (0,0,180), (180,255,255))
cv2.imshow("high_val", high_val)

background_mask = low_hue | high_hue | low_sat | high_sat | low_val | high_val
cv2.imshow("background_mask", background_mask)

coral = cv2.bitwise_not(background_mask)
cv2.imshow("not mask", coral)

if cv2.waitKey(0) == ord('s'):
    print("saving output as background_mask.jpg")
    cv2.imwrite("background_mask.jpg", background_mask)