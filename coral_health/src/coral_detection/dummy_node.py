from detect import *
import cv2

pt_string = "best_coral_detection.pt"
src_string = "dummy.jpg"

direct_image = cv2.imread("img39.jpg")

call_parser(pt_string, src_string, direct_image)