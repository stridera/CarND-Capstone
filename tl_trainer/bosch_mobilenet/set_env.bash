#!/bin/bash
# Must be called this way:
# . set_env.bash 
# So the env var are kept in the terminal sesion.
# 
# If the 1st line gives problems run in the terminal:
# sed -i -e 's/\r$//' scriptname.sh

export CURRENT_TRANSFER_LEARNING_FOLDER=/home/cricho/workspace/CarND-Capstone/tl_trainer/bosch_mobilenet
export OBJECT_DETECTION_API_FOLDER=/home/cricho/workspace/models
export BOSCH_DATA_FOLDER=/media/cricho/Data/machine_learning_data/carnd/bosch
