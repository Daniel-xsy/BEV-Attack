#!/usr/bin/env bash

CONFIG='projects/configs/bevformer/bevformer_base.py'
CHECKPOINT='../../checkpoints/BEVFormer/bevformer_r101_dcn_24ep.pth'
GPUS=2
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
