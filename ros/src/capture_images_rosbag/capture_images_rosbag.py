#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image

import numpy as np
from cv_bridge import CvBridge

import PIL.Image as PILImage


class ImageCapture(object):
    """
    """
    def __init__(self):
        rospy.init_node('capture_images_rosbag')

        self.path_dir = None          # Directory where to save the images, setup in the parameter server
        self.save_count = 0           # A counter for images used to generate appropriate name

        self.path_dir = rospy.get_param('~save_dir')
        rospy.Subscriber('/image_color', Image, self.image_cb)

        rospy.spin()

    def image_cb(self, msg):
        cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")[...,::-1]
        image = PILImage.fromarray(np.uint8(cv_image))

        filename = self.path_dir + str(self.save_count).zfill(5) + ".png"
        self.save_count += 1
        image.save(filename)


if __name__ == '__main__':
    try:
        ImageCapture()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic light detector based on tensorflow api capture.')
