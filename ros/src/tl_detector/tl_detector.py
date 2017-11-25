#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from styx_msgs.msg import Lane, TrafficLight
from std_msgs.msg import Int32

from cv_bridge import CvBridge
import tf as ros_tf

import numpy as np
import yaml
import math

import PIL.Image as PILImage
from light_classification.DoubleStageClassifier import DoubleStageClassifier

PI = math.pi
MAX_DIST = 100.0
MIN_DIST = 0.0
MAX_ANGLE = 15.0*PI/180.0


class TLDetector(object):
    """
    """
    def __init__(self):
        rospy.init_node('tl_detector')

        self.position = None          # Cartesian agent position (x, y)
        self.yaw = None               # Yaw of the agent
        self.stop_lines = None        # Waypoint coordinates correspondent to each traffic light [[x1, y1]..[xn, yn]]
        self.path_dir = None          # Directory where to save the images, setup in the parameter server
        self.base_waypoints = None
        self.closest_next_tl = -1     # Id of the closest traffic light. None if not detected
        self.stop_waypoint = None

        #self.model_file = rospy.get_param('~detection_model')
        #self.class_model_path = rospy.get_param('~classification_model_path')

        #TODO Generalize to different detectors
        self.TLDetector = DoubleStageClassifier()

        config_string = rospy.get_param("/traffic_light_config")
        config = yaml.load(config_string)
        self.stop_lines = np.array(config["stop_line_positions"])

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        self.tl_publisher = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        rospy.spin()

    def waypoints_cb(self, msg):
        """
        Callback storing all base track waypoints coordinates.
        :param msg: styx_msgs.msg.Lane type containing the array of base waypoints
        """
        base_waypoints = [np.array([p.pose.pose.position.x, p.pose.pose.position.y]) for p in msg.waypoints]
        self.base_waypoints = np.array(base_waypoints)

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
        self.yaw = ros_tf.transformations.euler_from_quaternion(quaternion)[2]
        if self.base_waypoints is not None:
            self.closest_next_tl = self._eval_next_closest_tl()

    def image_cb(self, msg):

        if self._eval_next_closest_tl() == -1:
            print "Image not processed, next traffic light too far away..."
            self.tl_publisher.publish(Int32(-1))
            return

        cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")[...,::-1]
        image = PILImage.fromarray(np.uint8(cv_image))
        image_np = np.expand_dims(image, axis=0)
        result = self.TLDetector.get_classification(image_np)

        if result == TrafficLight.RED:
            self._eval_stop_waypoint_index()
            self.tl_publisher.publish(Int32(self.stop_waypoint))
        elif result == TrafficLight.UNKNOWN:
            self.tl_publisher.publish(Int32(-2))
        else:
            self.tl_publisher.publish(Int32(-1))

    def _eval_stop_waypoint_index(self):
        if self.closest_next_tl >= 0:
            id_tl = self.closest_next_tl
            min_distance = 10000
            for k, wp in enumerate(self.base_waypoints):
                distance = TLDetector.eval_distance(self.stop_lines[id_tl][0], wp[0],
                                                    self.stop_lines[id_tl][1], wp[1])
                if distance < min_distance:
                    min_distance = distance
                    self.stop_waypoint = k
        else:
            self.stop_waypoint = -1

    def _eval_next_closest_tl(self):
        """
        Compares the location and yaw of the agent in respect to the traffic lights in the map.
        Looks for the following traffic light. If this exists and it is not too distant to the agent
        it returns the id of such traffic light. Otherwise it returns -1
        :return: The id of the next traffic light. None if non existing or too far.
        """
        if (self.stop_lines is not None) and (self.position is not None):
            for i, tl in enumerate(self.stop_lines):
                distance = TLDetector.eval_distance(tl[0], self.position[0], tl[1], self.position[1])
                direction = math.atan2( tl[1] - self.position[1] , tl[0] - self.position[0] )
                #https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
                angle_diff = math.atan2(math.sin(direction-self.yaw), math.cos(direction-self.yaw)) 
                print "angles..." , self.yaw*180/math.pi, direction*180/math.pi, angle_diff
                if (distance < MAX_DIST) and (distance > MIN_DIST) and (abs(angle_diff) < MAX_ANGLE) :
                    return i
        return -1

    @staticmethod
    def eval_distance(x1, x2, y1, y2):
        """
        Evaluates the Euclidean distance between points (x1, y1) and (x2, y2)
        :param x1: X-coordinate of first point
        :param x2: X-coordinate of second point
        :param y1: Y-coordinate of first point
        :param y2: Y-coordinate of second point
        :return: The Euclidean distance between the points
        """
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
