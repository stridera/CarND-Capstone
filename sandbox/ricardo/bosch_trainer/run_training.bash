#!/bin/bash

# Generate the temp.config with the proper paths to call the training.
temp_text='s#PATH_TO_BE_CONFIGURED#'$CURRENT_TRANSFER_LEARNING_FOLDER'#g'
sed -e ${temp_text} ${NETWORK_TO_BE_TRAINED}.config > temp.config

# Call the training function.
python ${OBJECT_DETECTION_API_FOLDER}/research/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${CURRENT_TRANSFER_LEARNING_FOLDER}/temp.config \
    --train_dir=${CURRENT_TRANSFER_LEARNING_FOLDER}/model_ckpt

