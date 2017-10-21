#!/bin/bash

folder='../../data/traffic-light-models'

wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz -P $folder
tar -xzvf $folder/ssd_mobilenet_v1_coco_11_06_2017.tar.gz -C $folder/
rm $folder/ssd_mobilenet_v1_coco_11_06_2017.tar.gz

wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz -P $folder
tar -xzvf $folder/ssd_inception_v2_coco_11_06_2017.tar.gz -C $folder/
rm $folder/ssd_inception_v2_coco_11_06_2017.tar.gz