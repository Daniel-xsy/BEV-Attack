# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Shaoyuan Xie
#  Used to visualize adversarial examples
# ---------------------------------------------

import os
import time
import numpy as np
import os.path as osp
import warnings
import argparse
from typing import Tuple,List

import mmcv
import torch

from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
import mmdet3d
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
import projects.mmdet3d_plugin
from projects.mmdet3d_plugin.attacks import build_attack

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import make_grid


def denormalize(img, mean, std):
    
    img = img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    img = img / 255

    return img


def show(imgs, mean, std):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = denormalize(img, mean, std)
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img)[:,:,::-1]) # [:,:,::-1]
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet attack a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--attack', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)
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
        shuffle=False
    )

    test_data = dataset[0]

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
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

    for n, p in model.named_parameters():
        p.requires_grad = False
    model = MMDataParallel(model, device_ids=[0])


    # Attack Debug
    if args.attack:

        attack_severity_type = cfg.attack_severity_type
        assert attack_severity_type in list(cfg.attack.keys()), f"Attack severity type {attack_severity_type} \
            is not a parameters in attack type {cfg.attack.type}"
        severity_list = cfg.attack[attack_severity_type]
        assert isinstance(severity_list, List), f"{attack_severity_type} in attack {cfg.attack.type} should be list\
            now {type(severity_list)}"
        cfg.attack[attack_severity_type] = np.random.choice(severity_list)
        print(f'Random choose {attack_severity_type}: {cfg.attack[attack_severity_type]}')

        attacker = build_attack(cfg.attack)

        # outputs = single_gpu_attack(model, data_loader, attacker)
        data_loader = iter(data_loader)
        data = next(data_loader)
        data = next(data_loader)
        
        mean = cfg.img_norm_cfg['mean']
        std = cfg.img_norm_cfg['std']

        if args.show:
            orig_img = make_grid(data['img'][0].data[0].squeeze())
            show(orig_img, mean, std)
            plt.savefig('original.png', dpi=200)
            plt.cla()

        print('running attacks')
        if attacker.is_train:
            attacker.train(model)
        inputs = attacker.run(model, **data)   

        if args.show:
            print('save results')
            adv_img = make_grid(inputs['img'][0].data[0].squeeze())
            show(adv_img, mean, std)
            plt.savefig('adver.png', dpi=200)
            plt.cla()

    else:
        data_loader = iter(data_loader)
        data = next(data_loader)
        if args.show:
            mean = cfg.img_norm_cfg['mean']
            std = cfg.img_norm_cfg['std']
            orig_img = make_grid(data['img'][0].data[0].squeeze()[0])
            show(orig_img, mean, std)
            plt.savefig('original.png', dpi=200)
            plt.cla()
        inputs = {'img': data['img'], 'img_metas': data['img_metas']}   
        results = model(return_loss=False, rescale=True, **inputs)
        




if __name__ == '__main__':
    main()
