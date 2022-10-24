#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/analysis_tools/debug_tool.py \
configs/attack/bevdepth-r50.py \
../../models/bevdepth/bevdepth-r50.pth \
--attack \

# -m debugpy --listen 5679 --wait-for-client 
# BEVDet
# configs/attack/bevdet-r50.py
# ../../models/bevdet/bevdet-r50.pth

# BEVDepth
# configs/attack/bevdepth-r50.py
# ../../models/bevdepth/bevdepth-r50.pth