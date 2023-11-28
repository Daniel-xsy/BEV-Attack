export PYTHONPATH="/root/autodl-tmp/models/BEV-Attack/zoo/BEVFormer":$PYTHONPATH 
export PYTHONPATH="/root/autodl-tmp/models/BEV-Attack/zoo/BEVFormer/tools":$PYTHONPATH 
export PYTHONPATH="/root/autodl-tmp/models/BEV-Attack/zoo/BEVFormer/tools/data_converter":$PYTHONPATH

python tools/create_data.py nuscenes \
--root-path /root/autodl-tmp/models/BEV-Attack/data/nuscenes \
--out-dir /root/autodl-tmp/models/BEV-Attack/data/nuscenes \
--extra-tag nuscenes \
--version v1.0-mini \
--canbus /root/autodl-tmp/models/BEV-Attack/data