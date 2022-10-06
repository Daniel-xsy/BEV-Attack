# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from mmcv.image import tensor2imgs
from os import path as osp

from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)


def single_gpu_attack(model,
                      data_loader,
                      attacker):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset

    # universal attack: train universial patch first
    if hasattr(attacker, 'train'):
        attacker.train(model, data_loader)
        pass

    a = dataset[0]
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):

        inputs = attacker.run(model, **data)     
        # inputs = {'img': data['img'], 'img_metas': data['img_metas']}   
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **inputs)

        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
