#!/usr/bin/env python
import rospy

from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import TrafficLightArray, TrafficLight
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


class ImageCapture(object):
    """
    Initializes the capture_images_sim node. This node subscribes to the current_pose, current_speed
    image_color and /vehicle/traffic_lights topics. For all these topics a throttled version is required
    in order to reduce publishing rate and avoid async issues.
    """
    def __init__(self):
        rospy.init_node('capture_images_sim')

        self.position = None          # Cartesian agent position (x, y)
        self.speed = None             # Cartesian speed components (vx, vy)
        self.yaw = None               # Yaw of the agent
        self.traffic_lights = None    # Waypoint coordinates correspondent to each traffic light [[x1, y1]..[xn, yn]]
        self.closest_next_tl = None   # Id of the closest traffic light. None if not detected
        self.state_closest_tl = -1    # State of the closest traffic light. -1 if not detected or too far away
        self.path_dir = None          # Directory where to save the images, setup in the parameter server
        self.save_count = 0           # A counter for images used to generate appropriate name
        #TODO Use the time stamp to generate the image name

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.speed_cb)
        rospy.Subscriber('/image_color_throttled', Image, self.image_cb)
        rospy.Subscriber('/vehicle/traffic_lights_throttled', TrafficLightArray, self.tl_cb)

        self.path_dir = rospy.get_param('~save_dir')
        config_string = rospy.get_param("/traffic_light_config")
        config = yaml.load(config_string)
        self.traffic_lights = np.array(config["stop_line_positions"])

        rospy.spin()

    def pose_cb(self, msg):
        """
        Callback function for current_pose topic. Extracts x, y and yaw values from the message and
        saves them in appropriate member variable. It also evaluates the id of the next traffic light.
        :param msg: Ros geometry_msgs/PoseStamped

        """
        position = msg.pose.position
        orientation = msg.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        self.position = (position.x, position.y)
        self.yaw = tf.transformations.euler_from_quaternion(quaternion)[2]
        self.closest_next_tl = self._eval_next_closest_tl()

    def speed_cb(self, msg):
        """
        Callback function for current_speed topic. It extracts the speed information and saves the x and y components
        in the correspondent member variable.
        :param msg: Ros geometry_msgs/TwistStamped
        """
        speed = msg.twist.linear
        self.speed = (speed.x, speed.y)

    def image_cb(self, msg):
        """
        Callback function for image_color topic. It transforms the message into a cv2 object. It reduces the
        image resolution (at the moment by half). Saves the message to the param server specified directory
        with customized name including an id, the traffic light state and the current speed in m/s
        :param msg: Ros message of type sensor_msgs/Image
        """
        cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        cv_image = ImageCapture._image_process(cv_image)

        path = self.path_dir + str(self.save_count).zfill(10) + "_"
        path += str(self.state_closest_tl) + "_"

        v = math.sqrt(self.speed[0] ** 2 + self.speed[1] ** 2)
        s_v = "%.2f" % v
        path += s_v + ".png"
        cv2.imwrite(path, cv_image)
        self.save_count += 1

    def tl_cb(self, msg):
        """
        Callback function for the traffic light topic. Sets the state of the following traffic light,
        given the pose (position and yaw) of the agent. If there is no close traffic light (or the next is
        too distant) then the state is set to -1.
        :param msg: Ros message carrying an array of the traffic light message as defined in ../styx_msgs/msg/
        """
        if self.closest_next_tl is not None:
            for tl in msg.lights:
                tl_x = tl.pose.pose.position.x
                tl_y = tl.pose.pose.position.y
                i = self.closest_next_tl
                distance = ImageCapture.eval_distance(tl_x, self.traffic_lights[i][0], tl_y, self.traffic_lights[i][1])
                DIST_TL_WP = 40
                # This value is fixed due to the different position between the traffic light and the associated
                # waypoint
                if distance < DIST_TL_WP:
                    self.state_closest_tl = tl.state
                    break
        else:
            self.state_closest_tl = -1

    @staticmethod
    def _image_process(im):
        """
        Applies a preprocessing on the image. At the moment only halves the resolution.
        :param im: Raw image
        :return: Preprocessed image
        """
        height, width = im.shape[:2]
        return  cv2.resize(im, (width / 2, height / 2), interpolation=cv2.INTER_CUBIC)

    def _eval_next_closest_tl(self):
        """
        Compares the location and yaw of the agent in respect to the traffic lights in the map.
        Looks for the following traffic light. If this exists and it is not too distant to the agent
        it returns the id of such traffic light. Otherwise it returns None
        :return: The id of the next traffic light. None if non existing or too far.
        """
        for i, tl in enumerate(self.traffic_lights):
            distance = ImageCapture.eval_distance(tl[0], self.position[0], tl[1], self.position[1])
            direction = math.atan2( tl[1] - self.position[1] , tl[0] - self.position[0] )
            if (distance < MAX_DIST) and (distance > MIN_DIST) and (abs(direction - self.yaw) < MAX_ANGLE) :
                return i
        return None

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
        ImageCapture()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start simulator traffic light images capture.')
