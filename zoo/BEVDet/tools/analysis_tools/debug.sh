#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/analysis_tools/debug_tool.py \
configs/attack/bevdet-r50.py \
/home/cixie/shaoyuan/BEV-Attack/models/bevdet/bevdet-r50.pth \
--out debug \

# -m debugpy --listen 5679 --wait-for-client 
# BEVDet
# configs/attack/bevdet-r50.py
# /home/cihangxie/shaoyuan/BEV-Attack/models/bevdet/bevdet-r50.pth

# BEVDepth
# configs/attack/bevdepth-r50.py
# /home/cihangxie/shaoyuan/BEV-Attack/models/bevdepth/bevdepth-r50.pth