#!/usr/bin/env python
import rospy

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image

import numpy as np
from cv_bridge import CvBridge
import tf
import cv2
import math
import yaml


PI = math.pi
MAX_DIST = 200.0
MIN_DIST = 0.0
MAX_ANGLE = 15.0*PI/180.0  #radians


class TLDetector(object):
    """
    """
    def __init__(self):
        rospy.init_node('tl_detector_tensorflow_api')

        self.position = None          # Cartesian agent position (x, y)
        self.yaw = None               # Yaw of the agent
        self.traffic_lights = None    # Waypoint coordinates correspondent to each traffic light [[x1, y1]..[xn, yn]]
        self.path_dir = None          # Directory where to save the images, setup in the parameter server
        self.save_count = 0           # A counter for images used to generate appropriate name


        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/image_color_throttled', Image, self.image_cb)

        self.path_dir = rospy.get_param('~save_dir')
        config_string = rospy.get_param("/traffic_light_config")
        config = yaml.load(config_string)
        self.traffic_lights = np.array(config["stop_line_positions"])

        rospy.spin()

    def pose_cb(self, msg):
        """
        Callback function for current_pose topic. Extracts x, y and yaw values from the message and
        saves them in appropriate member variable.
        :param msg: Ros geometry_msgs/PoseStamped

        """
        position = msg.pose.position
        orientation = msg.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        self.position = (position.x, position.y)
        self.yaw = tf.transformations.euler_from_quaternion(quaternion)[2]

    def image_cb(self, msg):
        pass




if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic light detector based on tensorflow api capture.')
