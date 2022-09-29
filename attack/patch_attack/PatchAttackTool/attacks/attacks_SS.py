'''
Giulio Rossolini
Semantic Segmentation Attacks
'''

import torch
import numpy as np

from torch.utils import data
from .attacks import Attack
import torch.nn.functional as F
import patch_utils as patch_utils


def norm(v):
    n = torch.norm(v, p=float('2'))
    return (v/n) if n > 0 else v # to avoid NaN error 

def sign(v):
    v = torch.sign(v)
    return v

GAMMA_VALUE = 1.0

class Attack_Untargeted_SS(Attack):

    def __init__(self, cfg, opt_obj):
        self.initStat()
        self.loss_adv =  get_adv_loss_function()
        self.gamma = GAMMA_VALUE
        super().__init__(cfg, opt_obj) 

    #------------------------------
    #Init attack stats 
    def initStat(self):
        self.epoch_loss_info_no_misc    = 0
        self.epoch_loss_info_misc       = 0
        self.epoch_loss_info_gamma      = 0
        self.epoch_loss_info_samples    = 0
        self.epoch_loss_info_smoothness = 0
        self.epoch_loss_info_NPS        = 0
        return

    #------------------------------
    # Print attack stats 
    def printStat(self, i, reset_stat = True):
        fmt_str = "Epochs [{:d}/{:d}]  Mean Losses: adv no misc {:.4f}, adv misc {:.4f}, Smoothing: {:.4f}, NPS {:.4f} (on {:d} training samples)  | gamma: {:.4f} "
        print_str = fmt_str.format(
                i + 1,
                self.cfg["adv_patch"]['patch_opt']["opt_iters"],
                self.epoch_loss_info_no_misc/self.epoch_loss_info_samples,
                self.epoch_loss_info_misc/self.epoch_loss_info_samples,
                self.epoch_loss_info_smoothness /self.epoch_loss_info_samples,
                self.epoch_loss_info_NPS /self.epoch_loss_info_samples,
                self.epoch_loss_info_samples,
                self.epoch_loss_info_gamma/self.epoch_loss_info_samples
            )
        print(print_str)
        i += 1

        if reset_stat:
            self.initStat()

        return

    #--------------------------
    # Label prepatching
    def label_prepatching(self, images, labels):
        return images, labels


    #--------------------------
    # Applied patch
    def applied_patches(self, images, intrinsic, extrinsic):
        perturbed_images, patch_masks = patch_utils.project_N_patches_batch(
                    images.clone(), 
                    self.opt_obj.model.patches, 
                    extrinsic, 
                    intrinsic,
                    self.opt_obj.patches_params_array,
                    device=self.opt_obj.device,
                    pixel_dim=self.opt_obj.pixel_width, offset=self.opt_obj.offset, rescale=self.opt_obj.rescale
        )
        return perturbed_images, patch_masks


    #--------------------------
    # Compute adv grad
    def compute_adv_grad(self):
        outputs = self.opt_obj.curr_outputs
        labels = self.opt_obj.curr_labels
        patch_masks = self.opt_obj.curr_patch_masks

        loss_no_misc, loss_misc, gamma  = self.loss_adv(input=outputs, target=labels, patch_mask = patch_masks.clone().to(self.opt_obj.device)) 
        self.epoch_loss_info_no_misc    += loss_no_misc.item()
        self.epoch_loss_info_misc       += loss_misc.item()
        self.epoch_loss_info_gamma      += gamma
        self.epoch_loss_info_samples    += outputs.shape[0]

        retain_graph_bool = True
        norm_grad_losses = []
        for _ in range(self.opt_obj.num_patches):
            norm_grad_losses.append([None,None])

        loss_array = [loss_no_misc, loss_misc]

        # compute and normalize each loss
        for count, l in enumerate(loss_array):
            self.opt_obj.optimizer.zero_grad()
            l.backward(retain_graph=retain_graph_bool)
            
            for patch_idx, p in enumerate(self.opt_obj.model.patches):
                if self.opt_obj.model.patches[patch_idx].grad is None:
                    return None
                grad_loss = self.opt_obj.model.patches[patch_idx].grad.data.clone().to(self.opt_obj.device)
                norm_grad_losses[patch_idx][count] = norm(grad_loss)

        adv_grad_patches = []
        for patch_idx, p in enumerate(self.opt_obj.model.patches):
            final_grad_adv = gamma * (-norm_grad_losses[patch_idx][0]) + (1-gamma) * (-norm_grad_losses[patch_idx][1])
            final_grad_adv = norm(final_grad_adv)
            adv_grad_patches.append(final_grad_adv)

        return adv_grad_patches



    
#-------------------------------------------------------
# Losses and related utils used during the attack optimization 
#-------------------------------------------------------
def get_adv_loss_function():
    return multi_scale_patch_composition

#------------------------------
# L = gamma * sum_{correct_pixels}(CE) + (1-gamma) * sum_{wrong_pixels}(CE)
# if gamma parameter = -1, a dynamic version of gamma is used
def untargeted_patch_composition(input, target, patch_mask, weight=None, size_average=True, gamma = 0.8):
    
    np, cp, hp, wp = patch_mask.size()

    n, c, h, w = input.size()

    # Handle inconsistent size between input and target label --> resize the target label
    # We assume that predicted labels are consistent a priori (only original label need to be resized)
    if len(list(target.shape)) > 1:
        nt, ht, wt = target.size()
        if h != ht and w != wt:  
            target = F.interpolate(target.float().unsqueeze(1), size=(h, w), mode="bilinear", align_corners=True).squeeze(1).long()

    # Handle inconsistent size between input and patch_mask --> resize the mask
    if h != hp and w != wp:  
        patch_mask = F.interpolate(patch_mask, size=(h, w), mode="bilinear", align_corners=True)
    
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    patch_mask = patch_mask.view(-1).detach().long()

    # cross entropy masks for both misclassified and not misclassified
    target_only_misc = target.clone()
    target_only_no_misc = target.clone()
    pred = torch.argmax(input, dim=1).to('cuda').detach()

    target_only_misc[target==pred]  = 250
    target_only_no_misc[target!=pred] = 250
    
    target_only_misc[patch_mask==1]  = 250
    target_only_no_misc[patch_mask==1] = 250

    del pred

    if gamma == -1:
        # compute a dynamic gamma value
        num_no_misclassified_pixels = torch.sum(target_only_no_misc!=250)
        num_total_pixels = target.size(0) - torch.sum(patch_mask)
        ret_gamma = num_no_misclassified_pixels/num_total_pixels
    else:
        ret_gamma = gamma

    
    if gamma == -2:
        # pixel-wise cross entropy on pixels out of patch
        target_without_patch = target.clone()
        target_without_patch[patch_mask==1] = 250
        loss_no_misc = F.cross_entropy(
        input, target_without_patch, reduction='sum', ignore_index=250, weight=weight
        )
        ret_gamma = 1.0
        del target_without_patch
        

    elif gamma == -3:
        # pixel-wise cross entropy on all image pixels
        loss_no_misc = F.cross_entropy(
        input, target, reduction='sum', ignore_index=250, weight=weight
        )
        ret_gamma = 1.0


    else:
        # loss for not yet misclassified elements
        loss_no_misc = F.cross_entropy(
            input, target_only_no_misc, reduction='sum', ignore_index=250, weight=weight
        )

    # loss for misclassified elements
    loss_misc = F.cross_entropy(
        input, target_only_misc, reduction='sum', ignore_index=250, weight=weight
    )

    del target_only_misc, target_only_no_misc
    return loss_no_misc, loss_misc, ret_gamma
    

#------------------------------
# Multi-input of the untargeted_patch_composition loss function
def multi_scale_patch_composition(input, target, weight=None, patch_mask = None, size_average=True, scale_weight=None, gamma = GAMMA_VALUE):
    if not isinstance(input, tuple):
        return untargeted_patch_composition(input=input, target=target, patch_mask = patch_mask, weight=weight, size_average=size_average, gamma=gamma)

    # Auxiliary weight
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 1.0 # > 1.0 means give more impotance to scaled outputs
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(input[0].device).detach()

    loss_no_misc, loss_misc, ret_gamma = 0, 0, None

    if not isinstance(target, tuple):
        target = [target.clone()] * len(input)

    for i,_ in enumerate(input):
        out_loss_no_misc, out_loss_misc, out_gamma  = untargeted_patch_composition(
            input=input[i], target=target[i], weight=weight, patch_mask = patch_mask, size_average=size_average, gamma=gamma)
        
        loss_no_misc  +=  scale_weight[i] * out_loss_no_misc
        loss_misc     +=  scale_weight[i] * out_loss_misc
        ret_gamma      =   out_gamma if (ret_gamma is None) else ret_gamma

    return loss_no_misc, loss_misc, ret_gamma



