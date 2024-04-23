#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/attack.py \
<config> \
<ckpt> \
--out <name> \

# patch_loc_vel_orie
# patch
# pgd_loc_vel_orie
# pgd_target_cls
# pgd


# -m debugpy --listen 5679 --wait-for-client 

# FCOS3D
# config = 'projects/configs/attack/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py'
# checkpoint_path = '../models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth'

# BEVFormer
# config = 'projects/configs/attack/bevformer_base_adv.py'
# checkpoint_path = '../models/bevformer/bevformer_r101_dcn_24ep.pth'

# BEVFormer small
# config = 'projects/configs/attack/bevformer_small_adv.py'
# checkpoint_path =  '../models/bevformer/bevformer_small_epoch_24.pth'

# DETR3D
# config = 'projects/configs/attack/detr3d_adv.py'
# checkpoint_path = '../models/detr3d/detr3d_resnet101_cbgs.pth'

# DETR3D_wo_CBGS
# config = 'projects/configs/attack/detr3d_wo_cbgs.py'
# checkpoint = '../models/detr3d/detr3d_resnet101.pth'

# PGD
# config = 'projects/configs/attack/pgd_r101_caffe_fpn_gn-head_2x16_1x_nus-mono3d.py'
# checkpoint_path = '../models/pgd/pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d_finetune_20211114_162135-5ec7c1cd.pth'

# PETR Vov
# config = 'projects/configs/attack/petr_vovnet_gridmask_p4_1600x640_train_cbgs.py'
# checkpoint_path = '../models/petr/petr-vov-p4-1600x640.pth'
