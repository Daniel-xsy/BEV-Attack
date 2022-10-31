#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m debugpy --listen 5679 --wait-for-client ./tools/analysis_tools/depth_estimation.py \
configs/attack/bevdet-r50.py \
../../models/bevdet/bevdet-r50.pth \
--show \
--attack \

# -m debugpy --listen 5679 --wait-for-client 
# BEVDet
# configs/attack/bevdet-r50.py
# ../../models/bevdet/bevdet-r50.pth

# BEVDepth
# configs/attack/bevdepth-r50.py
# ../../models/bevdepth/bevdepth-r50.pth