'''
Giulio Rossolini
Attack base class
'''

import os
import torch


def norm(v):
    n = torch.norm(v, p=float('2'))
    return (v/n) if n > 0 else v # to avoid NaN error 

def sign(v):
    v = torch.sign(v)
    return v

#--------------------------------------------------------------
# Attack base class
#--------------------------------------------------------------
class Attack(object):
    r"""
    Base class for the Attack
    """
    def __init__(self,cfg, opt_obj):
        self.cfg = cfg
        self.opt_obj = opt_obj
        self.smoothness_loss = smoothness_loss 
        self.NPS_loss = NPS

        # init NPS loss
        self.NPS_file = get_NPS_file(cfg)



    
    #-----------------------------
    # Compute physical grad 
    def compute_physical_grad(self):    
        smooth_loss = 0
        NPS_loss = 0
        for p in range(self.opt_obj.num_patches):
            smooth_loss += self.smoothness_loss(self.opt_obj.model.patches[p].to(self.opt_obj.device))  
            NPS_loss += self.NPS_loss(self.opt_obj.model.patches[p].to(self.opt_obj.device), patch_params=self.opt_obj.patches_params_array[p], color_list = self.NPS_file)

        self.epoch_loss_info_smoothness     += smooth_loss.data
        self.epoch_loss_info_NPS            += NPS_loss.data

        retain_graph_bool = True
        norm_grad_losses = []
        for _ in range(self.opt_obj.num_patches):
            norm_grad_losses.append([None,None])

        physical_grad_patches = []

        self.opt_obj.optimizer.zero_grad()
        smooth_loss.backward(retain_graph=retain_graph_bool)
        retain_graph_bool = False 
        # print(len(self.opt_obj.model.patches))
        for patch_idx, p in enumerate(self.opt_obj.model.patches):
            physical_grad_patches.append({'smooth':0, 'NPS': 0})
            grad_loss = self.opt_obj.model.patches[patch_idx].grad.data.clone().to(self.opt_obj.device)
            physical_grad_patches[patch_idx]['smooth'] = norm(grad_loss)

        return physical_grad_patches
    




#-------------------------------------------------------------
# Smoothness loss function
#-------------------------------------------------------------
def smoothness_loss(patch):
    device = patch.device.type
    p_h, p_w = patch.shape[-2:]
    # TODO Renormalize to avoid numerical problems
    if torch.max(patch) > 1:
        patch = patch / 255
    diff_w = torch.square(patch[:, :, :-1, :] - patch[:, :, 1:, :])
    zeros_w = torch.zeros((1, 3, 1, p_w), device=device)
    diff_h = torch.square(patch[:, :, :, :-1] - patch[:, :, :, 1:])
    zeros_h = torch.zeros((1, 3, p_h, 1), device=device)
    return torch.sum(torch.cat((diff_w, zeros_w), dim=2) + torch.cat((diff_h, zeros_h), dim=3))


#---------------------------------------------------------------------
# NON-PRINTABILITY SCORE 
#---------------------------------------------------------------------
def NPS(patch, patch_params, color_list=[]):
    device = patch.device.type
    color_list = color_list.to(device)
    p_h, p_w = patch.shape[-2:]
    
    mean = torch.Tensor(patch_params.set_loader.mean.reshape((1, 3, 1, 1))).to(device) #
    std = torch.Tensor(patch_params.set_loader.std.reshape((1, 3, 1, 1))).to(device) #
    max_val = 255
    color_max_val = 255
    if patch_params.set_loader.img_norm:
        max_val = 1

    patch = (patch * std + mean) / max_val
    color_list = color_list / color_max_val
    diff_col = torch.sub(patch, color_list)
    diff_norm = torch.norm(diff_col, dim=1)
    diff_prod = torch.prod(diff_norm.reshape((-1, p_w * p_h)), dim=0)
    return torch.sum(diff_prod)



#-------------------------------------------------------------------
# NPS utils
#-------------------------------------------------------------------
def get_NPS_file(cfg):
    NPS_cfg = cfg['adv_patch']['patch_opt']['loss']['NPS']
    NPS_name = NPS_cfg['name']
    NPS_args = NPS_cfg["args"]
    P = []
    assert os.path.isfile(NPS_args)
    with open(NPS_args, "r") as f:
        lines = f.readlines()
        for line in lines:
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
    return P
