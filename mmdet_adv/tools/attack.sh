#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=1 \
python attack.python \
'/home/cihangxie/shaoyuan/BEV-Attack/mmdet_adv/projects/configs/attack/pgd_r101_caffe_fpn_gn-head_2x16_1x_nus-mono3d.py' \
'/home/cihangxie/shaoyuan/BEV-Attack/models/pgd/pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d_finetune_20211114_162135-5ec7c1cd.pth' \

# FCOS3D
# config = '/home/cihangxie/shaoyuan/BEV-Attack/mmdet_adv/projects/configs/attack/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py'
# checkpoint_path = '/home/cihangxie/shaoyuan/BEV-Attack/models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth'

# BEVFormer
# config = '/home/cihangxie/shaoyuan/BEV-Attack/mmdet_adv/projects/configs/attack/bevformer_base_adv.py'
# checkpoint_path = '/home/cihangxie/shaoyuan/BEV-Attack/models/bevformer/bevformer_r101_dcn_24ep.pth'

# DETR3D
# config = '/home/cihangxie/shaoyuan/BEV-Attack/mmdet_adv/projects/configs/attack/detr3d_adv.py'
# checkpoint_path = '/home/cihangxie/shaoyuan/BEV-Attack/models/detr3d/detr3d_resnet101_cbgs.pth'

# PGD
# config = '/home/cihangxie/shaoyuan/BEV-Attack/mmdet_adv/projects/configs/attack/pgd_r101_caffe_fpn_gn-head_2x16_1x_nus-mono3d.py'
# checkpoint_path = '/home/cihangxie/shaoyuan/BEV-Attack/models/pgd/pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d_finetune_20211114_162135-5ec7c1cd.pth'