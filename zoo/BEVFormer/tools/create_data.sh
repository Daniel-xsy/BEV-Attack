PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/create_data.py nuscenes \
--root-path /data1/shaoyuan/nuscenes \
--out-dir /data1/shaoyuan/nuscenes \
--extra-tag nuscenes \
--version v1.0-mini \
--canbus /data1/shaoyuan \