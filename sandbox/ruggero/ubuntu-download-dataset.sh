#!/bin/bash

folder='../../data/traffic-light-models'

net1='ssd_mobilenet_v1_coco_11_06_2017.tar.gz'
net2='ssd_inception_v2_coco_11_06_2017.tar.gz'
net3='rfcn_resnet101_coco_11_06_2017.tar.gz'


if [ ! -d "$folder/ssd_mobilenet_v1_coco_11_06_2017" ]; then
    wget http://download.tensorflow.org/models/object_detection/$net1 -P $folder
    tar -xzvf $folder/$net1 -C $folder/
    rm $folder/$net1
else
    echo "SSD mobilenet already downloaded...skipping..."
fi

if [ ! -d "$folder/ssd_mobilenet_v1_coco_11_06_2017" ]; then
    wget http://download.tensorflow.org/models/object_detection/$net2 -P $folder
    tar -xzvf $folder/$net2 -C $folder/
    rm $folder/$net2
else
    echo "SSD inception already downloaded...skipping..."
fi

if [ ! -d "$folder/rfcn_resnet101_coco_11_06_2017" ]; then
    wget http://download.tensorflow.org/models/object_detection/$net3 -P $folder
    tar -xzvf $folder/$net3 -C $folder/
    rm $folder/$net3
else
    echo "Region Based Resnet already downloaded...skipping..."
fi