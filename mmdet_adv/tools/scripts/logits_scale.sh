PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m debugpy --listen 5678 --wait-for-client ./tools/analysis_tools/logits_scale.py \
/home/cixie/shaoyuan/BEV-Attack/mmdet_adv/projects/configs/attack/detr3d_adv.py \
/home/cixie/shaoyuan/BEV-Attack/mmdet_adv/logits/detr3d_adv_results.pkl \