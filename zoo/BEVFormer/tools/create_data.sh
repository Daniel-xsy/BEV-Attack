PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/create_data.py nuscenes \
--root-path /data2/shaoyuan/nuscenes \
--out-dir /data2/shaoyuan/nuscenes \
--extra-tag nuscenes \
--version v1.0 \
--canbus /data2/shaoyuan \