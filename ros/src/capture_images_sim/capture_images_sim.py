#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf
import cv2
import yaml


class ImageCapture(object):
    def __init__(self):

        self.pose = None
        self.traffic_lights = []
        self.image = None

        sub_current_pose = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub_image = rospy.Subscriber('/image_color', Image, self.image_cb)
        suc_tl = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.tl_cb)
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

    def pose_cb(self, msg):
        pass

    def image_cb(self, msg):
        pass

    def tl_cb(self, msg):
        pass

if __name__ == '__main__':
    try:
        ImageCapture()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start simulator traffic light images capture.')
