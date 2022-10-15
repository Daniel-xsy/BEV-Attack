#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m debugpy --listen 5679 --wait-for-client ./tools/attack.py \
'/home/cixie/shaoyuan/BEV-Attack/zoo/BEVDet/configs/bevdepth/bevdepth-r50.py' \
'/home/cixie/shaoyuan/BEV-Attack/models/bevdepth/bevdepth-r50.pth' \
--out patch_attack15x15 \