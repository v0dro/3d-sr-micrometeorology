#!/bin/bash

ROOT_DIR=/Users/sameer/gitrepos/3d-sr-micrometeorology

IMAGE_PATH="${ROOT_DIR}/pytorch.sif"
SCRIPT_PATH="${ROOT_DIR}/pytorch/script/train_model.py"
CONFIG_PATH="${ROOT_DIR}/pytorch/config/default.yml"

echo "image path = ${IMAGE_PATH}"
echo "script path = ${SCRIPT_PATH}"
echo "config path = ${CONFIG_PATH}"

export PYTHONPATH=$ROOT_DIR/pytorch

# singularity exec \
#   --nv \
#   --env PYTHONPATH=$ROOT_DIR/pytorch \
#   ${IMAGE_PATH} python3 ${SCRIPT_PATH} --config_path ${CONFIG_PATH} --world_size 1

python3 ${SCRIPT_PATH} --config_path ${CONFIG_PATH} --world_size 1
