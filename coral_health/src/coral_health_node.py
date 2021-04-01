#!/usr/bin/env python

import rospy

# https://git.epoxsea.com/rov-2019/cannon_length/blob/master/src/nodes/cannon_length_node.py
from sensor_msgs.msg import CompressedImage
from std_msgs/msg import Int16
# from rov_messages.msg import __

# Setup proper filepaths
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# importing the class definition
from coral_health.program import coral_image

def main():
    rospy.init_node("coral_health_node", anonymous=False)
