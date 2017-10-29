#!/usr/bin/env python
import os

import rospy
from sensor_msgs.msg import Image

import numpy as np
from cv_bridge import CvBridge

import PIL.Image as PILImage


class ImageCapture(object):
    """
    Manages the "capture_images_rosbag" node, used to subscribe to the image_color topic and
    to save the received images into the ~save_dir directory specified in the param server.
    """
    def __init__(self):
        rospy.init_node('capture_images_rosbag')

        self.path_dir = None          # Directory where to save the images, setup in the parameter server
        self.save_count = 0           # A counter for images used to generate appropriate saving name

        self.path_dir = rospy.get_param('~save_dir')
        self._dir_management()

        rospy.Subscriber('/image_color', Image, self.image_cb)
        rospy.spin()

    def image_cb(self, msg):
        """
        Callback function for the image_color topic. It reads the Image message, converts it
        into a PILImage type and saves the image in a specific directory with customized
        name.
        :param msg: ROS sensor_msgs/Image message type
        """
        cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")[...,::-1]
        image = PILImage.fromarray(np.uint8(cv_image))

        filename = self.path_dir + str(self.save_count).zfill(5) + ".png"
        self.save_count += 1
        image.save(filename)

    def _dir_management(self):
        """
        Checks if the saving directory exists, otherwise it creates it
        """
        if not os.path.isdir(self.path_dir):
            os.makedirs(self.path_dir)


if __name__ == '__main__':
    try:
        ImageCapture()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start rosbag site images capture.')
