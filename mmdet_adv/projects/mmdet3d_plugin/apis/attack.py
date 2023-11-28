# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
import mmcv
import torch
from os import path as osp


def single_gpu_attack(model,
                      wb_model,
                      data_loader,
                      attacker):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        wb_model (nn.Module): White box model to be attacked
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    wb_model.eval()
    results = []
    dataset = data_loader.dataset

    # universal attack: train universial patch first
    if hasattr(attacker, 'loader'):
        if attacker.is_train:
            attacker.train(wb_model)
            mmcv.dump(attacker.patches, f'./uni_patch/{wb_model.module.__class__.__name__}_coslr_size{attacker.patch_size[0]}_scale{attacker.scale}_lr{attacker.lr}_sample{attacker.max_train_samples}.pkl')

    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):

        # if i < 10:
        # inputs = {'img': data['img'], 'img_metas': data['img_metas']}   
        # else:
        inputs = attacker.run(wb_model, **data)    
        # inputs = {'img': data['img'], 'img_metas': data['img_metas']}   
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **inputs)

        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def custom_collect_img(data):
    data = deepcopy(data)
    img_ = data['img'][0]
    for i in range(len(img_)):
        img_.data[i] = img_.data[i].squeeze()

    return data
