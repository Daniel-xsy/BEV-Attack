# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
import os
from typing import List
import os.path as osp
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.attacks import build_attack
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor

from shutil import copyfile

from tools.utils import single_gpu_attack

from tools.analysis_tools.parse_results import collect_metric, Logging_str

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet attack a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
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

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # set random seeds
    set_random_seed(0, deterministic=False)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    dataset.test_mode = False
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # # only used for debug
    # a = dataset[0]

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # if args.fuse_conv_bn:
    #     model = fuse_conv_bn(model)
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


    attacker = build_attack(cfg.attack)
    if hasattr(attacker, 'loader'):
        attack_dataset = build_dataset(attacker.loader)
        attack_loader = build_dataloader(
            attack_dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        )
        attacker.loader = attack_loader

    assert distributed==False, "Attack not support distributed"

    model = MMDataParallel(model, device_ids=[0])

    attack_severity_type = cfg.attack_severity_type
    assert attack_severity_type in list(cfg.attack.keys()), f"Attack severity type {attack_severity_type} \
        is not a parameters in attack type {cfg.attack.type}"
    severity_list = cfg.attack[attack_severity_type]
    assert isinstance(severity_list, List), f"{attack_severity_type} in attack {cfg.attack.type} should be list\
        now {type(severity_list)}"
        
    logging = Logging_str(osp.join('log', cfg.model.type, args.out, f"{os.path.splitext(os.path.basename(args.config))[0]}.md"))
    logging.write(f"## Model Configuration\n")
    logging.write(f"```")
    logging.write(cfg.pretty_text)
    logging.write(f"```\n")

    for i in range(len(severity_list)):
        cfg.attack[attack_severity_type] = severity_list[i]
        # build attack
        attacker = build_attack(cfg.attack)
        if hasattr(attacker, 'loader'):
            attack_dataset = build_dataset(attacker.loader)
            attack_loader = build_dataloader(
                attack_dataset,
                samples_per_gpu=samples_per_gpu,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=False,
                shuffle=False,
                nonshuffler_sampler=cfg.data.nonshuffler_sampler,
            )
            attacker.loader = attack_loader

        outputs = single_gpu_attack(model, data_loader, attacker)


        kwargs = {}
        kwargs['jsonfile_prefix'] = osp.join('results', cfg.model.type, args.out, f"{attack_severity_type}_{severity_list[i]}")

        if not osp.isdir(kwargs['jsonfile_prefix']): os.makedirs(kwargs['jsonfile_prefix'])
        # copy config file
        copyfile(args.config, osp.join(kwargs['jsonfile_prefix'], 'config.py'))

        eval_kwargs = cfg.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric='bbox', **kwargs))

        results = dataset.evaluate(outputs, **eval_kwargs)
        logging.write(f"### {attack_severity_type} {severity_list[i]}\n")
        collect_metric(results, logging)


if __name__ == '__main__':
    main()
