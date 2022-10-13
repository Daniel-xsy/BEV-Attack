PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python  -m debugpy --listen 5679 --wait-for-client ./exps/mv/bev_depth_lss_r50_256x704_128x128_24e_2key.py \
--ckpt_path /home/cihangxie/shaoyuan/BEV-Attack/models/bevdepth/bev_depth_lss_r50_256x704_128x128_24e_2key.pth \
-e -b 1 --gpus 1 \
# 