# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

try:
  OBJECT_DETECTION_API_FOLDER = os.environ['OBJECT_DETECTION_API_FOLDER']
  TRANSFER_LEARNING_FOLDER = os.environ['CURRENT_TRANSFER_LEARNING_FOLDER']
except:
  sys.exit('ERROR: Run set_env.bash to have the folders information as environmental variables.')
  
try:  
  LAST_CHECKPOINT = os.environ['LAST_CHECKPOINT']
  print('Last checkpoint: ', LAST_CHECKPOINT)
except:
  sys.exit('ERROR: LAST_CHECKPOINT environmental variable not defined')

sys.path.insert(0, OBJECT_DETECTION_API_FOLDER)

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

if tf.__version__ != '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
TRAIN_PATH           = TRANSFER_LEARNING_FOLDER + '/model_ckpt'
PATH_TO_FROZEN_GRAPH = TRANSFER_LEARNING_FOLDER + '/model_frozen_graph_' + LAST_CHECKPOINT + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(TRANSFER_LEARNING_FOLDER, 'bosch_label_map.pbtxt')
NUM_CLASSES = 2

# Folder with the images in jpg
PATH_TO_TEST_IMAGES_DIR = TRANSFER_LEARNING_FOLDER + '/test_images'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, file) for file in os.listdir(PATH_TO_TEST_IMAGES_DIR) if file.endswith('.png')]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
IMAGE_SIZE = (20, 14)


if( not os.path.isfile(PATH_TO_FROZEN_GRAPH) ):
  sys.exit('The file you are trying to use doesn''t exist: ' + PATH_TO_FROZEN_GRAPH)

 
print('Reading graph ...')    
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
    
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print(categories)
print(category_index)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)



with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    
    print('Starting inference of {} images ...'.format(len(TEST_IMAGE_PATHS)))
    for im_num, image_path in enumerate(TEST_IMAGE_PATHS):
      

      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=4)
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)
      plt.title(image_path)    
      #print('boxes', boxes)
      #print('classes', classes)
      #print('scores', scores)

      plt.show()

print('End of the program')    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    