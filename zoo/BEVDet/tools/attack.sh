#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m debugpy --listen 5679 --wait-for-client ./tools/attack.py \
'configs/attack/bevdet-r50.py' \
'/home/cihangxie/shaoyuan/BEV-Attack/models/bevdet/bevdet-r50.pth' \
--out patch_attack15x15 \

# -m debugpy --listen 5679 --wait-for-client 

# BEVDet
# configs/attack/bevdet-r50.py
# /home/cihangxie/shaoyuan/BEV-Attack/models/bevdet/bevdet-r50.pth

# BEVDepth
# configs/attack/bevdepth-r50.py
# /home/cihangxie/shaoyuan/BEV-Attack/models/bevdepth/bevdepth-r50.pth