#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/analysis_tools/debug_tool.py \
configs/attack/bevdet-r50.py \
../../models/bevdepth/bevdepth-r50.pth \
--out attack \

# -m debugpy --listen 5679 --wait-for-client 
# BEVDet
# configs/attack/bevdet-r50.py
# ../../models/bevdet/bevdet-r50.pth

# BEVDepth
# configs/attack/bevdepth-r50.py
# ../../models/bevdepth/bevdepth-r50.pth