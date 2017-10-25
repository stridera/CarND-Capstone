#!/usr/bin/env python
import rospy

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image

import numpy as np
from cv_bridge import CvBridge
import tf
import tensorflow
import cv2
import math
import yaml

import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import PIL.Image as PILImage


class TLDetector(object):
    """
    """
    def __init__(self):
        rospy.init_node('tl_detector_tf_api')

        self.position = None          # Cartesian agent position (x, y)
        self.yaw = None               # Yaw of the agent
        self.traffic_lights = None    # Waypoint coordinates correspondent to each traffic light [[x1, y1]..[xn, yn]]
        self.path_dir = None          # Directory where to save the images, setup in the parameter server
        self.save_count = 0           # A counter for images used to generate appropriate name

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/image_color_throttled', Image, self.image_cb)

        self.path_dir = rospy.get_param('~save_dir')
        self.model_file = rospy.get_param('~model_file')

        self.detection_graph = tensorflow.Graph()
        self._import_tf_graph()


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
        cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        image = PILImage.fromarray(np.uint8(cv_image))
        #TODO Some kind of preprocessing
        with self.detection_graph.as_default():
            with tensorflow.Session(graph=self.detection_graph) as sess:
                image_np = np.expand_dims(image, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                                    feed_dict={image_tensor: image_np})

                for i in range(boxes.shape[1]):
                    if scores[0, i] > 0.5 and classes[0,i] == 10:
                        TLDetector.draw_bounding_box_on_image(image, boxes[0, i, 0], boxes[0, i, 1], boxes[0, i, 2],
                                                              boxes[0, i, 3], color='red', thickness=4,
                                                              display_str_list=(), use_normalized_coordinates=True)
                filename = self.path_dir + str(self.save_count).zfill(5) + ".png"
                self.save_count += 1
                image.save(filename)

    def _import_tf_graph(self):
        with self.detection_graph.as_default():
            od_graph_def = tensorflow.GraphDef()
            with tensorflow.gfile.GFile(self.model_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tensorflow.import_graph_def(od_graph_def, name='')

    #TODO restyle method to minimal requirements (taken from tensorflow)
    @staticmethod
    def draw_bounding_box_on_image(image,
                                   ymin,
                                   xmin,
                                   ymax,
                                   xmax,
                                   color='red',
                                   thickness=4,
                                   display_str_list=(),
                                   use_normalized_coordinates=True):

        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        if use_normalized_coordinates:
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
        else:
            (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=thickness, fill=color)
        try:
            font = ImageFont.truetype('arial.ttf', 24)
        except IOError:
            font = ImageFont.load_default()

        text_bottom = top
        # Reverse list and print from bottom to top.
        for display_str in display_str_list[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle(
                [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                                  text_bottom)],
                fill=color)
            draw.text(
                (left + margin, text_bottom - text_height - margin),
                display_str,
                fill='black',
                font=font)
            text_bottom -= text_height - 2 * margin


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic light detector based on tensorflow api capture.')
