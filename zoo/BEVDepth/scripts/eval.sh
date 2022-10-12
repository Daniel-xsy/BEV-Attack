PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./exps/fusion/bev_depth_fusion_lss_r50_256x704_128x128_24e.py \
--ckpt_path /home/cihangxie/shaoyuan/BEV-Attack/models/bevdepth/bev_depth_lss_r50_256x704_128x128_24e_2key.pth \
-e -b 8 --gpus 8 \