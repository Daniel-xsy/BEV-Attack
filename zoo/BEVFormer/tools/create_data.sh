PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/create_data.py nuscenes \
--root-path /data1/data/shaoyuan/nuscenes \
--out-dir /data1/data/shaoyuan/nuscenes \
--extra-tag nuscenes \
--version v1.0 \
--canbus /data1/data/shaoyuan \