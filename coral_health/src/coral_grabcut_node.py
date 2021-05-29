#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()

from sensor_msgs.msg import CompressedImage
from advtrn_msg.msg import VideoStream

# Import class definitions from ./scripts/coral_health_program.py
# from scripts.detect import yolo inference program
from scripts.coral_grabcut_program import coral_grabcut
from coral_detection.detect import *

cb_index = 0

def main_callback(msg, callback_args):
    global cb_index
    cb_index += 1
    frame_number = cb_index
    frame = msg                 # for compressedImage to be published

    # if cb_index%5 != 0:
    #     return

    rospy.loginfo("Received frame number: {}".format(frame_number))

    """ Step 0: Reading Image Inputs """
    try:
        np_arr = np.frombuffer(frame.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except CvBridgeError as e:
        print(e)




    """ Step 1: Background Removal with Grabcut """
    # TODO:
    rect = call_parser("best_coral_detection.pt", "dummy.jpg", cv_image)    # rect in format [x0, y0, x1, y1]
    rect_x0y0wh = (rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1])      # rect in format (x0, y0, w, h)
    new_coral = coral_grabcut(cv_image, rect_x0y0wh)
    grabcut_output = new_coral.grabcut()

    cv2.imshow("Orig Video Feed", cv_image)
    cv2.imshow("OUTPUT grabcut", grabcut_output)
    cv2.waitKey(1)






    """ Final Step: Publish image to GUI """
    # result_image = grabcut_output

    # result_publisher = callback_args
    # result_image = bridge.cv2_to_compressed_imgmsg(result_image)
    # result_publisher.publish(result_image)
    # rospy.loginfo("Published frame number: {}".format(cb_index))



    # Newline prettyprint
    print("")



def main():
    rospy.init_node("coral_grabcut_node", anonymous=False)
    rospy.loginfo("coral_grabcut_node initialized")

    result_publisher = rospy.Publisher("/coral/result", CompressedImage, queue_size = 5)
    # rospy.Subscriber("camera/Coral_image/compressed", CompressedImage, main_callback, callback_args=result_publisher)
    rospy.Subscriber('advtrn/VideoStream', VideoStream, main_callback)

    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down from user interrupt...")
            rospy.loginfo("Shutting down from user interrupt...")
            rospy.signal_shutdown()
            break

if __name__ == "__main__":
    main()
