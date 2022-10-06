#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=1 \
python -m debugpy --listen 30021 --wait-for-client $(dirname "$0")/test.py ./projects/configs/bevformer/bevformer_base.py /home/cihangxie/shaoyuan/BEV-Attack/models/bevformer/bevformer_r101_dcn_24ep.pth  --eval bbox