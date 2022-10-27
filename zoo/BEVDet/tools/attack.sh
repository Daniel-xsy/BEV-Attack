#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/attack.py \
configs/attack/bevdet-r50.py \
../../models/bevdet/bevdet-r50.pth \
--out patch_loc_vel_orie \

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