#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/attack.py \
configs/attack/bevdepth-r50.py \
/home/cihangxie/shaoyuan/BEV-Attack/models/bevdepth/bevdepth-r50.pth \
--out PGD \

# -m debugpy --listen 5678 --wait-for-client 

# BEVDet
# configs/attack/bevdet-r50.py
# /home/cihangxie/shaoyuan/BEV-Attack/models/bevdet/bevdet-r50.pth

# BEVDepth
# configs/attack/bevdepth-r50.py
# /home/cihangxie/shaoyuan/BEV-Attack/models/bevdepth/bevdepth-r50.pth