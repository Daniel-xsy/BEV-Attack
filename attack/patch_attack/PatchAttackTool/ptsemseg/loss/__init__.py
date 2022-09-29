import logging
import functools
import torch
import os

from ptsemseg.loss.loss import (
    cross_entropy2d,
    bootstrapped_cross_entropy2d,
    multi_scale_cross_entropy2d,
    multi_scale_patch_composition,
    multi_scale_patch_composition_targeted,
    smoothness_loss,
    NPS,
)


logger = logging.getLogger("ptsemseg")

key2loss = {
    "cross_entropy": cross_entropy2d,
    "bootstrapped_cross_entropy": bootstrapped_cross_entropy2d,
    "multi_scale_cross_entropy": multi_scale_cross_entropy2d,
    "multi_scale_patch_composition": multi_scale_patch_composition,
    "multi_scale_patch_composition_targeted": multi_scale_patch_composition_targeted,
    "smoothness_loss": smoothness_loss,
    "NPS": NPS,
}


def get_loss_function(cfg_training):
    if cfg_training["loss"] is None:
        logger.info("Using default cross entropy loss")
        return cross_entropy2d

    else:
        loss_dict = cfg_training["loss"]
        loss_name = loss_dict["name"]
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}
        loss_params['weight'] = torch.Tensor([1, 0.01, 0.01, 1, 1, 1, 1, 1, 1, 0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).to('cuda')

        if loss_name not in key2loss:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))

        logger.info("Using {} with {} params".format(loss_name, loss_params))
        return functools.partial(key2loss[loss_name], **loss_params)
    
def get_patch_loss_function(cfg_training):
    adv_loss = cfg_training["loss"]["adv_loss"]
    smooth_loss = cfg_training["loss"]["smoothness"]
    NPS = cfg_training["loss"]["NPS"]
    
    adv_loss_name = adv_loss["name"]
    adv_loss_arg = adv_loss["args"]
    
    smooth_loss_name = smooth_loss["name"]
    smooth_loss_args = smooth_loss["args"]
    
    NPS_name = NPS["name"]
    NPS_args = NPS["args"]
    P = []
    assert os.path.isfile(NPS_args)
    with open(NPS_args, "r") as f:
        lines = f.readlines()
        for line in lines:
#             print(line)
            split_str = line.split(',')
            val_r = split_str[0].strip()
            if '(' in val_r:
                val_r = val_r[-1]
            val_g = split_str[1].strip()
            val_b = split_str[2].strip()
            if ')' in val_b:
                val_b = val_b[0]
            P.append([float(val_r), float(val_g), float(val_b)])
    
            
    P = torch.Tensor(P).reshape((-1, 3, 1, 1))
    
    weights = (adv_loss["mult_factor"], smooth_loss["mult_factor"], NPS["mult_factor"])
    
    if adv_loss_name not in key2loss:
        raise NotImplementedError("Loss {} not implemented".format(adv_loss_name))
    if smooth_loss_name not in key2loss:
        raise NotImplementedError("Loss {} not implemented".format(smooth_loss_name))
    if NPS_name not in key2loss:
        raise NotImplementedError("Loss {} not implemented".format(NPS_name))
        
    losses_tuple = (
        _init_adv_loss(adv_loss_name, adv_loss_arg), 
        functools.partial(key2loss[smooth_loss_name]), 
        functools.partial(key2loss[NPS_name], color_list=P), 
                   )
    return losses_tuple, weights

def _init_adv_loss(loss_name, arg):
    if loss_name in ("cross_entropy", "bootstrapped_cross_entropy", "multi_scale_cross_entropy") or arg is None:
        return functools.partial(key2loss[loss_name])
    elif loss_name in ("multi_scale_patch_composition","multi_scale_patch_composition_targeted",) and arg is not None:
        return functools.partial(key2loss[loss_name], gamma=arg)
    
    
    
