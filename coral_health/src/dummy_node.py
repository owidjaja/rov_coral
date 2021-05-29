import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/oscar/catkin_ws/src/coral_health/src/coral_detection/')
from coral_detection.detect import *

import cv2

# src = cv2.imread("./coral_detection/img39.jpg")

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture("test_video.mp4")
frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    frame_num += 1

    if frame_num % 10 != 0:
        continue
    elif not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv2.imwrite("temp_frame.jpg", frame)
    call_parser("/home/oscar/catkin_ws/src/coral_health/src/coral_detection/best_coral_detection.pt", "/home/oscar/catkin_ws/src/coral_health/src/temp_frame.jpg")

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()