#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/attack.py \
./configs/attack/bevdet-r101.py \
../../checkpoints/BEVDet/bevdet-r101.pth \
--out autopgd \

# patch_loc_vel_orie
# pgd_loc_vel_orie
# pgd_target_cls
# pgd

# -m debugpy --listen 5678 --wait-for-client 

# BEVDet
# configs/attack/bevdet-r50.py
# ../../models/bevdet/bevdet-r50.pth

# BEVDepth
# configs/attack/bevdepth-r50.py
# ../../models/bevdepth/bevdepth-r50.pth