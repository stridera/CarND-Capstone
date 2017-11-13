#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from styx_msgs.msg import Lane
from std_msgs.msg import Int32

from cv_bridge import CvBridge
import tf as ros_tf

import numpy as np
import yaml
import os
import math
import time

import cv2
import PIL.Image as PILImage

import tensorflow as tf
from keras.models import model_from_json


PI = math.pi
MAX_DIST = 150.0
MIN_DIST = 0.0
MAX_ANGLE = 15.0*PI/180.0  #radians


class TLDetector(object):
    """
    """
    def __init__(self):
        rospy.init_node('tl_detector_tf_api')

        self.position = None          # Cartesian agent position (x, y)
        self.yaw = None               # Yaw of the agent
        self.stop_lines = None        # Waypoint coordinates correspondent to each traffic light [[x1, y1]..[xn, yn]]
        self.path_dir = None          # Directory where to save the images, setup in the parameter server
        self.base_waypoints = None
        self.closest_next_tl = -1     # Id of the closest traffic light. None if not detected
        self.stop_waypoint = None

        self.model_file = rospy.get_param('~detection_model')
        self.Detector = Detector(self.model_file)

        self.class_model_path = rospy.get_param('~classification_model_path')
        self.Classifier = Classifier(self.class_model_path)

        config_string = rospy.get_param("/traffic_light_config")
        config = yaml.load(config_string)
        self.stop_lines = np.array(config["stop_line_positions"])

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/image_color_throttled', Image, self.image_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
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

        time_detection_start = time.time()
        (boxes, scores, classes, num_detections) = self.Detector.detect(image_np)
        time_detection_end = time.time()
        print('Detection inference: {:0.3f} ms'.format((time_detection_end - time_detection_start) * 1000.0))

        tl_detections = [i for i in range(boxes.shape[1]) if (scores[0, i] > 0.8 and classes[0,i] == 10)]
        if len(tl_detections) == 0:
            print ("No traffic light detected")
            return None
        else:
            print ("Detected possible {} traffic lights".format(len(tl_detections)))

        cropped = np.array([self._prepare_for_class(image_np, boxes[:, i, :]) for i in tl_detections if i is not None])

        if len(cropped) == 0:
            print ("Detected no traffic lights...")
            self.tl_publisher.publish(Int32(-1))
            return

        classification = self.Classifier.classify(cropped)
        print ("Next traffic light state: {}".format(classification))

        if classification == 0:
            self._eval_stop_waypoint_index()
            self.tl_publisher.publish(Int32(self.stop_waypoint))
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

        for i, tl in enumerate(self.stop_lines):
            distance = TLDetector.eval_distance(tl[0], self.position[0], tl[1], self.position[1])
            direction = math.atan2( tl[1] - self.position[1] , tl[0] - self.position[0] )
            if (distance < MAX_DIST) and (distance > MIN_DIST) and (abs(direction - self.yaw) < MAX_ANGLE) :
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

    @staticmethod
    def _prepare_for_class(image, box):
        """
        It crops a box from the image
        :param image: The image
        :param box: The box to crop
        :return: The resized cropped image or None if the conditions are not satisfied
        """
        shape = image.shape
        (left, right, top, bottom) = (box[0, 1] * shape[2], box[0, 3] * shape[2],
                                      box[0, 0] * shape[1], box[0, 2] * shape[1])

        #Assuming that crop_height > crop_width valid for tf_api and standard traffic lights
        crop_height = int(bottom - top)
        crop_width = int(right - left)

        if 1.5*crop_width < crop_height < 3.5*crop_width:
            center = (int(left)+ int(right)) // 2
            if (center - (crop_height // 2) < 0):
                cropped = image[0, int(top): int(bottom), 0: crop_height , :]
            elif (center + (crop_height // 2) > shape[2]):
                cropped = image[0, int(top): int(bottom), shape[2] - crop_height: shape[2], :]
            else:
                cropped = image[0, int(top) : int(bottom), center - (crop_height // 2): center + (crop_height//2), :]
            resized = cv2.resize(cropped, (50, 50), interpolation = cv2.INTER_CUBIC)
            return resized[...,::-1]
        else:
            return None




class Detector(object):
    def __init__(self, model_file):

        self.model_file = model_file
        self.detection_graph = tf.Graph()
        self._import_tf_graph()
        self.sess = tf.Session(graph=self.detection_graph)

    def _import_tf_graph(self):
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def detect(self, image_np):
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        return self.sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np})

class Classifier(object):
    def __init__(self, model_path):
        json_file = open(os.path.join(model_path, 'model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.classification_model = model_from_json(loaded_model_json)
        self.classification_model.load_weights(os.path.join(model_path, 'model.h5'))

        self.classification_model._make_predict_function()  # see https://github.com/fchollet/keras/issues/6124
        self.classification_graph = tf.get_default_graph()

    def classify(self, cropped):
        with self.classification_graph.as_default():
            predictions = self.classification_model.predict(cropped)

        results = []
        for i,p in enumerate(predictions):
            if np.max(p) > 0.9:
                result = np.argmax(p)
                results.append(result)
        if len(results) == 0:
            return None
        else:
            counts = np.bincount(results)
            if len(counts[counts == np.max(counts)]) == 1:
                return np.argmax(counts)
            else:
                return None



if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic light detector based on tensorflow api capture.')
