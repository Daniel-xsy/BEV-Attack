import os
import argparse

import mmcv
import torch

from mmcv import Config

from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
import mmdet3d
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from projects.mmdet3d_plugin.datasets.builder import build_dataloader

from mmdet.core.bbox.builder import BBOX_ASSIGNERS


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet attack a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('results', help='checkpoint file')
    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    cfg = Config.fromfile(args.config)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                if isinstance(cfg.plugin_dir, str):
                    cfg.plugin_dir = [cfg.plugin_dir]
                # import multi plugin modules
                for plugin_dir_ in cfg.plugin_dir:
                    _module_dir = os.path.dirname(plugin_dir_)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]

                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    set_random_seed(0, deterministic=False)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    # test_mode false to return ground truth used in the attack
    # chnage after build the dataset to avoid data filtering
    # an ugly workaround to set test_mode = False
    dataset.test_mode = False
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    assigner = BBOX_ASSIGNERS.build(cfg.attack.assigner)
    results = mmcv.load(args.results)

    for batch_id, data in enumerate(data_loader):
        gt_bboxes_3d = data['gt_bboxes_3d']
        gt_labels_3d = data['gt_labels_3d']
        outputs = [results[batch_id]]
        assigner.assign(outputs, gt_bboxes_3d, gt_labels_3d)
        a = 1


if __name__ == '__main__':
    main()
