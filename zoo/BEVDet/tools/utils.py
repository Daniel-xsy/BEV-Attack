# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
import mmcv
import torch


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
    if hasattr(attacker, 'loader'):
        attacker.train(model)

    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):

        inputs = attacker.run(model, **data)
        # inputs = {'img_inputs': data['img_inputs'], 'img_metas': data['img_metas']}   
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
