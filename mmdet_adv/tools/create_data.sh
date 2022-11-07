PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/create_data.py nuscenes \
--root-path ../nuscenes \
--out-dir ../nuscenes \
--extra-tag nuscenes \
--version v1.0 \
--canbus /data2/shaoyuan \