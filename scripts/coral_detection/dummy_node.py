from detect import *
import cv2

pt_string = "best_coral_detection.pt"
src_string = "img39.jpg"

direct_image = cv2.imread(src_string)

call_parser(pt_string, src_string, direct_image)