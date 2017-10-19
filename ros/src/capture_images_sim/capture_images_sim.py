#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, TwistStamped, Quaternion
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge
import tf
import cv2
import math
import yaml

PI = math.pi

MAX_DIST = 150.0
MIN_DIST = 5.0
MAX_ANGLE = 15.0*PI/180.0  #radians

class ImageCapture(object):
    def __init__(self):
        rospy.init_node('capture_images_sim')

        self.position = None
        self.yaw = None
        self.traffic_lights = None
        self.image = None
        self.closest_next_tl = None   # The index in the traffic light config list
        self.state_closest_tl = None
        self.path_dir = None

        sub_current_pose = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub_image = rospy.Subscriber('/image_color', Image, self.image_cb)
        suc_tl = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.tl_cb)

        self.path_dir = rospy.get_param('~save_dir')

        config_string = rospy.get_param("/traffic_light_config")
        config = yaml.load(config_string)
        self.traffic_lights = np.array(config["stop_line_positions"])

        rospy.spin()

    def pose_cb(self, msg):
        position = msg.pose.position
        orientation = msg.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        self.position = (position.x, position.y)
        self.yaw = tf.transformations.euler_from_quaternion(quaternion)[2]

        #final_distance = None
        #final_orientation = None
        for i, tl in enumerate(self.traffic_lights):
            distance = ImageCapture.eval_distance(tl[0], self.position[0], tl[1], self.position[1])
            direction = math.atan2( tl[1] - self.position[1] , tl[0] - self.position[0] )
            if (distance < MAX_DIST) and (distance > MIN_DIST) and (abs(direction - self.yaw) < MAX_ANGLE) :
                #final_distance = distance
                #final_orientation = direction*180.0/PI
                self.closest_next_tl = i
                break
            else:
                self.closest_next_tl = None
        #print self.closest_next_tl, final_distance, final_orientation, self.position, self.yaw*180/PI

    def image_cb(self, msg):
        cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        #if self.state_closest_tl is not None:
        #    print "saving image", self.state_closest_tl

    def tl_cb(self, msg):
        if self.closest_next_tl is not None:
            for tf in msg.lights:
                tf_x = tf.pose.pose.position.x
                tf_y = tf.pose.pose.position.y
                i = self.closest_next_tl
                distance = ImageCapture.eval_distance(tf_x, self.traffic_lights[i][0], tf_y, self.traffic_lights[i][1])
            #TODO Set 10 to a parameter to change
                if distance < 10:
                    self.state_closest_tl = tf.state
                    break
        else:
            self.state_closest_tl = None

        print self.state_closest_tl

    @staticmethod
    def eval_distance(x1, x2, y1, y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)


if __name__ == '__main__':
    try:
        ImageCapture()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start simulator traffic light images capture.')
