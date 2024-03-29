#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/analysis_tools/debug_tool.py \
projects/configs/attack/petr_r50dcn_gridmask_p4.py \
../models/petr/petr-r50-p4-1408x512.pth \
--show \

# -m debugpy --listen 5679 --wait-for-client 

# BEVDepth 
# config = 'projects/configs/attack/bevdepth-r50.py'
# checkpoint_path = '../models/bevdepth/bevdepth-r50.pth'

# FCOS3D
# config = 'projects/configs/attack/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py'
# checkpoint_path = '../models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth'

# BEVFormer
# config = 'projects/configs/attack/bevformer_base_adv.py'
# checkpoint_path = '../models/bevformer/bevformer_r101_dcn_24ep.pth'

# BEVFormer_small
# config = 'projects/configs/attack/bevformer_small_adv.py'
# checkpoint_path =  '../models/bevformer/bevformer_small_epoch_24.pth'

# BEVFormer_tiny
# config = '/home/cixie/shaoyuan/BEV-Attack/mmdet_adv/projects/configs/attack/bevformer_tiny.py'
# checkpoint_path = '../models/bevformer/bevformer_tiny_epoch_24.pth'

# DETR3D
# config = 'projects/configs/attack/detr3d_adv.py'
# checkpoint_path = '../models/detr3d/detr3d_resnet101_cbgs.pth'

# DETR3D_wo_CBGS
# config = 'projects/configs/attack/detr3d_wo_cbgs.py'
# checkpoint = '../models/detr3d/detr3d_resnet101.pth'

# PGD
# config = 'projects/configs/attack/pgd_r101_caffe_fpn_gn-head_2x16_1x_nus-mono3d.py'
# checkpoint_path = '../models/pgd/pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d_finetune_20211114_162135-5ec7c1cd.pth'

# PETR
# config = 'projects/configs/attack/petr_r50dcn_gridmask_p4.py'
# checkpoint_path = '../models/petr/petr-r50-p4-1408x512.pth'