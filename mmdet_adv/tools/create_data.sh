PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/create_data.py nuscenes \
--root-path /data1/data/shaoyuan/nuscenes_mini \
--out-dir /data1/data/shaoyuan/nuscenes_mini \
--extra-tag nuscenes \
--version v1.0-mini \
--canbus /data1/data/shaoyuan \