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
import os

import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import PIL.Image as PILImage

import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.models import model_from_json

import time

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

        self.path_dir = rospy.get_param('~save_dir')
        self.model_file = rospy.get_param('~model_file')

        self.detection_graph = tensorflow.Graph()
        self._import_tf_graph()

        self.sess = tensorflow.Session(graph=self.detection_graph)

        self.class_model_path = rospy.get_param('~class_model_path')
        self.classification_model = None
        self._import_keras_model()

        config_string = rospy.get_param("/traffic_light_config")
        config = yaml.load(config_string)
        self.traffic_lights = np.array(config["stop_line_positions"])

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/image_color_throttled', Image, self.image_cb)

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
        cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")[...,::-1]
        image = PILImage.fromarray(np.uint8(cv_image))
        #TODO Some kind of preprocessing

        image_np = np.expand_dims(image, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        time_detection_start = time.time()

        (boxes, scores, classes, num_detections) = self.sess.run([boxes, scores, classes, num_detections],
                                                            feed_dict={image_tensor: image_np})

        time_detection_end = time.time()

        print('Detection inference: {:0.3f} ms'.format((time_detection_end - time_detection_start)   * 1000.0))

        traffic_light_detections = 0

        cropped = []
        for i in range(boxes.shape[1]):
            if scores[0, i] > 0.5 and classes[0,i] == 10:
                TLDetector.draw_bounding_box_on_image(image, boxes[0, i, 0], boxes[0, i, 1], boxes[0, i, 2],
                                                      boxes[0, i, 3], color='red', thickness=4,
                                                      display_str_list=(), use_normalized_coordinates=True)
                traffic_light_detections += 1
                cropped.append(self._prepare_for_class(image_np, boxes[:,i,:]))
        cropped = np.array(cropped)

        predictions = self.classification_model.predict(cropped)
        print (predictions)


        filename = self.path_dir + str(self.save_count).zfill(5) + ".png"
        self.save_count += 1
        #image.save(filename)
        #cv2.imwrite(filename, cropped[0])

        if traffic_light_detections > 0:
            print('Detected {:d} traffic light{}'.format(traffic_light_detections, 's' if traffic_light_detections > 1 else ''))


    def _prepare_for_class(self, image, boxes):
        shape = image.shape
        (left, right, top, bottom) = (boxes[0, 1] * shape[2], boxes[0, 3] * shape[2],
                                      boxes[0, 0] * shape[1], boxes[0, 2] * shape[1])
        crop_height = int(bottom - top)
        crop_width = int(right - left)
        #TODO Consider more complicated cases
        #if crop_height > crop_width:
        center = (int(left)+ int(right)) // 2
        cropped = image[0, int(top) : int(bottom), center - (crop_height // 2): center + (crop_height//2), :]
        resized = cv2.resize(cropped, (50, 50), interpolation = cv2.INTER_CUBIC)
        return resized



    #    shape = image.shape
    #    if shape[1] > shape[2]


    def _import_tf_graph(self):
        with self.detection_graph.as_default():
            od_graph_def = tensorflow.GraphDef()
            with tensorflow.gfile.GFile(self.model_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tensorflow.import_graph_def(od_graph_def, name='')

    def _import_keras_model(self):

        #input_shape = (50, 50, 3)
        #num_classes = 3

        #self.classification_model.add(Convolution2D(32, kernel_size=(2, 2), padding='same',
        #                                            activation='relu', input_shape=input_shape))
        #self.classification_model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.classification_model.add(Convolution2D(64, (2, 2), activation='relu', padding='same'))
        #self.classification_model.add(MaxPooling2D(pool_size=(2, 2)))
        #self.classification_model.add(Dropout(0.25))
        #self.classification_model.add(Flatten())
        #self.classification_model.add(Dense(128, activation='relu'))
        #self.classification_model.add(Dropout(0.5))
        #self.classification_model.add(Dense(num_classes, activation='softmax'))

        json_file = open(os.path.join(self.class_model_path, 'model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.classification_model = model_from_json(loaded_model_json)
        self.classification_model.load_weights(os.path.join(self.class_model_path, 'model.h5'))

        self.classification_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                             metrics=['accuracy'])






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
