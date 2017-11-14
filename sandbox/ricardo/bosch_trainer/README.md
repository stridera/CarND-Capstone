
# Before tranining

## Object detection API

Install the object detecion API from Google and make it work. Follow the steps here:

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

And clone the repo:

git clone https://github.com/tensorflow/models/tree/master/research/object_detection

## Bosch dataset

Download the dataset from:

https://hci.iwr.uni-heidelberg.de/node/6132

I used the RGB version of the train data (~6 Gb).

Convert the data to TFRecord (needed for the API):

Set environment: Open the file set_env.bash and put the correct folders. Run it as follows to keep them as environmental variables during the terminal sesion:

. set_env.bash

Run the following:

python create_bosch_traffic_light_tf_record.py

## Get the pretrained network.

There are several here:

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

For some reason, the new pretrained networks (they updated recently) were not working, but the linkst to the old ones (they work for me) are still working:

The one that I have been using (small):

http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz

The one that we may want to use:

http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz

Put the files in the folder /model_ckpt of your model training folder.

# Configure the training

Edit the .config to set the proper hyperparameters. 

Do not modify the PATH_TO_BE_CONFIGURED, as it will be taken care by the launching scripts (below).

## Steps

### Set environment

Open the file set_env.bash and put the correct folders. Run it as follows to keep them as environmental variables during the terminal sesion:

. set_env.bash

### Training

Run the following command (in principle it should not matter where you are).

./run_training.bash

During training, to check the GPU state, in a new terminal:

nvidia-smi --loop-ms=1000

If your GPU has plenty of memory free, you can try increasing "batch_size" in the .config file.

Also, tensorboard can be used to check progress with:

tensorboard --logdir=${CURRENT_TRANSFER_LEARNING_FOLDER}/model_ckpt

(if it is a different terminal, remember to run set_env.bash)

Then open in a browser:

http://localhost:6006/


### Generate frozen graph file

One the training is finished, or just stopped in order to check how it is working, run:

. run_generate_frozen_graph.bash

NOTE: For the moment, the file must be edited manually setting the variable LAST_CHECKPOINT to the highest checkpoint saved (or the one to be converted).

### Test on some samples.

Put figures in the folder /test_figures and run:

python test_current_checkpoint.py 

It will generate matplotlib plots to have a quick look.
