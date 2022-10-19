#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/attack.py \
projects/configs/attack/bevformer_tiny.py \
/home/cixie/shaoyuan/BEV-Attack/models/bevformer/bevformer_tiny_epoch_24.pth \
--out PGD \

# -m debugpy --listen 5679 --wait-for-client 
# BEVDepth 
# config = 'projects/configs/attack/bevdepth-r50.py'
# checkpoint_path = '/home/cixie/shaoyuan/BEV-Attack/models/bevdepth/bevdepth-r50.pth'

# FCOS3D
# config = 'projects/configs/attack/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py'
# checkpoint_path = '/home/cihangxie/shaoyuan/BEV-Attack/models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth'

# BEVFormer
# config = 'projects/configs/attack/bevformer_base_adv.py'
# checkpoint_path = '/home/cihangxie/shaoyuan/BEV-Attack/models/bevformer/bevformer_r101_dcn_24ep.pth'

# BEVFormer tiny
# config = '/home/cixie/shaoyuan/BEV-Attack/mmdet_adv/projects/configs/attack/bevformer_tiny.py'
# checkpoint_path = '/home/cixie/shaoyuan/BEV-Attack/models/bevformer/bevformer_tiny_epoch_24.pth'

# DETR3D
# config = 'projects/configs/attack/detr3d_adv.py'
# checkpoint_path = '/home/cihangxie/shaoyuan/BEV-Attack/models/detr3d/detr3d_resnet101_cbgs.pth'

# PGD
# config = 'projects/configs/attack/pgd_r101_caffe_fpn_gn-head_2x16_1x_nus-mono3d.py'
# checkpoint_path = '/home/cihangxie/shaoyuan/BEV-Attack/models/pgd/pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d_finetune_20211114_162135-5ec7c1cd.pth'