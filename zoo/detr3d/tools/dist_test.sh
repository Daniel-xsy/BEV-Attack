#!/usr/bin/env bash

CONFIG=./projects/configs/detr3d/detr3d_res101_gridmask.py
CHECKPOINT=/root/autodl-tmp/models/BEV-Attack/checkpoints/DETR/detr3d_resnet101.pth
GPUS=2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
