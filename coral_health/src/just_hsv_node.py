#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()

from sensor_msgs.msg import CompressedImage
# from rov_messages.msg import __

# Import class definition from ./scripts/coral_health_program.py
from scripts.coral_health_program import coral_image

pw_hsv_arr = None
cb_index = 0

def main_callback(msg, callback_args):
    global first_new, pw_hsv_arr, cb_index
    cb_index += 1
    old = callback_args[0]
    frame_number = cb_index
    frame = msg     # for compressedImage

    rospy.loginfo("Received frame number: {}".format(frame_number))

    """ Step 0: Reading Image Inputs """
    try:
        np_arr = np.frombuffer(frame.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except CvBridgeError as e:
        print(e)

    new = coral_image(cv_image, [13,50,50])
  
    """ Step 1: Background Removal """
    if (pw_hsv_arr == None):
        # print("FIRST TIME INIT PW_HSV_ARR")
        pw_hsv_arr = new.background_remover(False, waitKeyTime=0)
    else:
        # print("HAS HSV ALREADY")
        _ = new.background_remover(True, pw_hsv_arr, waitKeyTime=1)

    # cv2.imshow("Old Mask in Main...", old.pink_white_mask)
    # cv2.imshow("New Mask in Main...", new.pink_white_mask)
    # cv2.destroyAllWindows()
    # cv2.destroyWindow("hsv")
    # cv2.destroyWindow("Trackbar_Window")

    oldAND = cv2.bitwise_and(old.src, old.src, mask=old.pink_white_mask)
    newAND = cv2.bitwise_and(new.src, new.src, mask=new.pink_white_mask)

    cv2.imshow("Orig Video Feed", cv_image)
    cv2.imshow("Masked Video Feed", newAND)
    cv2.waitKey(1)

    result_publisher = callback_args[1]
    result_image = bridge.cv2_to_compressed_imgmsg(newAND)
    result_publisher.publish(result_image)
    rospy.loginfo("Published frame number: {}".format(cb_index))

    # rospy.loginfo("Background Removal OK")
    print("")

def main():
    rospy.init_node("just_hsv_node", anonymous=False)
    rospy.loginfo("just_hsv_node initialized")

    """ TODO: CHANGE ABSOLUTE PATH ACCORDING TO CONTROL BOX """
    old_src = cv2.imread("/home/oscar/catkin_ws/src/coral_health/src/old_scaled40.jpg")
    if old_src is None:
        old_src = cv2.imread("/home/hammerhead/catkin_ws/src/coral_health/resources/old_scaled40.jpg")
    old = coral_image(old_src, [30,50,50])
    # old.background_remover(False, waitKeyTime)
    # old.alignment()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    result_publisher = rospy.Publisher("/coral/result", CompressedImage, queue_size = 10)
    rospy.Subscriber("camera/Coral_image/compressed", CompressedImage, main_callback, callback_args=[old, result_publisher])

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
