#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()

from sensor_msgs.msg import CompressedImage

# Import class definition from ./scripts/coral_health_program.py
from scripts.coral_grabcut_program import coral_grabcut

cb_index = 0

def main_callback(msg, callback_args):
    global cb_index
    cb_index += 1
    frame_number = cb_index
    frame = msg                 # for compressedImage to be published

    rospy.loginfo("Received frame number: {}".format(frame_number))

    """ Step 0: Reading Image Inputs """
    try:
        np_arr = np.frombuffer(frame.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except CvBridgeError as e:
        print(e)

    """ Step 1: Background Removal with Grabcut """

    new_coral = coral_grabcut(cv_image)
    grabcut_output = new_coral.grabcut()

    cv2.imshow("Orig Video Feed", cv_image)
    cv2.imshow("OUTPUT grabcut", grabcut_output)
    cv2.waitKey(1)

    """ Final Step: Publish image to GUI """
    out_image = grabcut_output

    result_publisher = callback_args
    result_image = bridge.cv2_to_compressed_imgmsg(result_image)
    result_publisher.publish(result_image)
    rospy.loginfo("Published frame number: {}".format(cb_index))

    # Newline prettyprint
    print("")



def main():
    rospy.init_node("coral_grabcut_node", anonymous=False)
    rospy.loginfo("coral_grabcut_node initialized")

    result_publisher = rospy.Publisher("/coral/result", CompressedImage, queue_size = 5)
    rospy.Subscriber("camera/Coral_image/compressed", CompressedImage, main_callback, callback_args=result_publisher)

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
