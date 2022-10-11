#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=2,3 \
python -m debugpy --listen 30021 --wait-for-client $(dirname "$0")/test.py \
/home/cixie/shaoyuan/BEV-Attack/mmdet_adv/projects/configs/attack/bevformer_base_adv.py \
/home/cihangxie/shaoyuan/BEV-Attack/models/bevformer/bevformer_r101_dcn_24ep.pth  \
--eval bbox