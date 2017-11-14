#!/bin/bash

export LAST_CHECKPOINT=23386

python ${OBJECT_DETECTION_API_FOLDER}/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path      ${CURRENT_TRANSFER_LEARNING_FOLDER}/temp.config \
    --trained_checkpoint_prefix ${CURRENT_TRANSFER_LEARNING_FOLDER}/model_ckpt/model.ckpt-${LAST_CHECKPOINT} \
    --output_directory          ${CURRENT_TRANSFER_LEARNING_FOLDER}/model_frozen_graph_${LAST_CHECKPOINT}

# Delete the temp.config file.
rm temp.config
