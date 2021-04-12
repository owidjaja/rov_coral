#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()

""" DENNIS """
from sensor_msgs.msg import CompressedImage
# from rov_messages.msg import __

""" TEMP MINH CODE """
from advtrn_msg.msg import VideoStream

# Import class definition from ./scripts/coral_health_program.py
from scripts.coral_health_program import coral_image

first_new = True
pw_hsv_arr = None

def main_callback(msg, callback_args):
    global first_new, pw_hsv_arr

    old = callback_args[0]

    frame_number = msg.frame_number.data    # dependent on advtrn_msg::VideoStream
    rospy.loginfo("Received frame number: {}".format(frame_number))

    # Only process every 10 image
    if (frame_number % 10 != 0):
        return
    rospy.loginfo("Processing frame number: {}".format(frame_number))

    """ Step 0: Reading Image Inputs """
    # old_src = cv2.imread("/home/oscar/catkin_ws/src/coral_health/src/old_scaled40.jpg")
    # old = coral_image(old_src, [30,50,50])

    try:
        cv_image = bridge.imgmsg_to_cv2(msg.frame, "bgr8")
    except CvBridgeError as e:
        print(e)

    new = coral_image(cv_image, [13,50,50])

    cv2.imshow("old_src", old.src)
    rospy.loginfo("Images successfully read")

    """ Step 1: Background Removal """

    print("HERE FIRST_NEW IS {}".format(first_new))
    # old.background_remover()
    if (first_new == True):
        pw_hsv_arr = new.background_remover(False)
        first_new = False
    else:
        print("HAS HSV ALREADY")
        _ = new.background_remover(True, pw_hsv_arr)

    cv2.imshow("Old Mask in Main...", old.pink_white_mask)
    cv2.imshow("New Mask in Main...", new.pink_white_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    oldAND = cv2.bitwise_and(old.src, old.src, mask=old.pink_white_mask)
    newAND = cv2.bitwise_and(new.src, new.src, mask=new.pink_white_mask)

    rospy.loginfo("Background Removal OK")

    """ Step 2: Alignment of the two images """

    # old.alignment()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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

    """ Processing old image once """
    old_src = cv2.imread("/home/oscar/catkin_ws/src/coral_health/src/old_scaled40.jpg")
    old = coral_image(old_src, [30,50,50])
    old.background_remover(False)
    old.alignment()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    first_new = True
    pw_hsv_arr = None

    """ temporarily working with Minhs code"""
    rospy.Subscriber("advtrn/VideoStream", VideoStream, main_callback, callback_args=[old])
    # rospy.Subscriber("camera/Coral_Image", compressed, main_callback)
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
