#!/usr/bin/env python3

import rospy
import cv2

""" TO BE CHANGED ACCORDING TO DENNIS """
# https://git.epoxsea.com/rov-2019/cannon_length/blob/master/src/nodes/cannon_length_node.py
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int16
# from rov_messages.msg import __

""" TEMP MINH CODE """
from advtrn_msg.msg import VideoStream

from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()

# Setup proper filepaths
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# importing the class definition
from scripts.coral_health_program import coral_image

""" Processing old image once """
# old = coral_image("old_scaled40.jpg", [30,50,50], "/home/oscar/catkin_ws/src/coral_health/src/")
# old.background_remover()

# put main code under a func to be callback
# rospy spin once
def main_callback(msg):
    print("looping in main_cb")

    """ Step 0: Reading Image Inputs """
    old_src = cv2.imread("/home/oscar/catkin_ws/src/coral_health/src/old_scaled40.jpg")
    old = coral_image(old_src, [30,50,50])

    try:
        cv_image = bridge.imgmsg_to_cv2(msg.frame, "bgr8")
    except CvBridgeError as e:
        print(e)

    new = coral_image(cv_image, [13,50,50])

    cv2.imshow("old_src", old.src)
    rospy.loginfo("Images successfully read")

    """ Step 1: Background Removal """

    old.background_remover()
    new.background_remover()

    cv2.imshow("Old Mask in Main...", old.pink_white_mask)
    cv2.imshow("New Mask in Main...", new.pink_white_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    oldAND = cv2.bitwise_and(old.src, old.src, mask=old.pink_white_mask)
    newAND = cv2.bitwise_and(new.src, new.src, mask=new.pink_white_mask)

    rospy.loginfo("Background Removal OK")

    """ Step 2: Alignment of the two images """

    old.alignment()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    new.alignment()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("Old Aligned Mask in Main...", old.pers_transformed)
    cv2.imshow("New Aligned Mask in Main...", new.pers_transformed)

    rospy.loginfo("Image alignment OK")

    """ Step 3: Identify changes """

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    rospy.init_node("coral_health_node", anonymous=False)
    rospy.loginfo("coral_health_node initialized")

    """ temporarily working with Minhs code"""
    rospy.Subscriber("advtrn/VideoStream", VideoStream, main_callback)
    print("After subscribing")

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
