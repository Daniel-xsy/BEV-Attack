import sys
sys.path.append('/home/cihangxie/shaoyuan/BEV-Attack')
sys.path.append('/home/cihangxie/shaoyuan/BEV-Attack/zoo/BEVFormer')
import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp

config = '/home/cihangxie/shaoyuan/BEV-Attack/zoo/BEVFormer/projects/configs/bevformer/bevformer_base.py'
checkpoint = '/home/cihangxie/shaoyuan/BEV-Attack/models/bevformer/bevformer_r101_dcn_24ep.pth'

cfg = Config.fromfile(config)

if cfg.get('custom_imports', None):
    from mmcv.utils import import_modules_from_strings
    import_modules_from_strings(**cfg['custom_imports'])

# import modules from plguin/xx, registry will be updated
if hasattr(cfg, 'plugin'):
    if cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]

            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)
        else:
            # import dir is the dirpath for the config file
            _module_dir = os.path.dirname(config)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)

samples_per_gpu = 1
# build the dataloader
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False,
    nonshuffler_sampler=cfg.data.nonshuffler_sampler,
)

# build the model and load checkpoint
cfg.model.train_cfg = None
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
fp16_cfg = cfg.get('fp16', None)
if fp16_cfg is not None:
    wrap_fp16_model(model)
checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')

if 'CLASSES' in checkpoint.get('meta', {}):
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    model.CLASSES = dataset.CLASSES
# palette for visualization in segmentation tasks
if 'PALETTE' in checkpoint.get('meta', {}):
    model.PALETTE = checkpoint['meta']['PALETTE']
elif hasattr(dataset, 'PALETTE'):
    # segmentation dataset has `PALETTE` attribute
    model.PALETTE = dataset.PALETTE

data = dataset[0]