import torch
import torch.nn.functional as F

#---------------------------------------------------------------------
# Modified loss function for patch optimization. L = gamma * sum_{correct_pixels}(CE) + (1-gamma) * sum_{wrong_pixels}(CE)
# if gamma parameter = -1, a fixed dynamic version of gamma is used:
# gamma = num_no_misclassified_pixels/num_total_pixels
#---------------------------------------------------------------------
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
    

#---------------------------------------------------------------------
# Multi-input of the untargeted_patch_composition loss function (to consider also aux_logits)
#---------------------------------------------------------------------
def multi_scale_patch_composition(input, target, weight=None, patch_mask = None, size_average=True, scale_weight=None, gamma = 0.9):
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
#         color_max_val = 255
    
    patch = (patch * std + mean) / max_val
    color_list = color_list / color_max_val
#     print(patch.shape, color_list.shape)
#     print(color_list)
    diff_col = torch.sub(patch, color_list)
#     print(diff_col)
#     print(diff_col.shape)
    diff_norm = torch.norm(diff_col, dim=1)
#     print(diff_norm.shape)
    diff_prod = torch.prod(diff_norm.reshape((-1, p_w * p_h)), dim=0)
#     print(diff_prod.shape)
    
    return torch.sum(diff_prod)



#---------------------------------------------------------------------
# Smoothness loss function
#---------------------------------------------------------------------
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
#---------------------------------------------------------------------
def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)

    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250, reduction='mean'
    )
    return loss



#---------------------------------------------------------------------
#---------------------------------------------------------------------
def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None):
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(target.device)

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss



#---------------------------------------------------------------------
#---------------------------------------------------------------------
def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, reduce=False, size_average=False, ignore_index=250
        )

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)




#---------------------------------------------------------------------
# Modified loss function for patch optimization. L = gamma * sum_{correct_pixels}(CE) + (1-gamma) * sum_{wrong_pixels}(CE)
# if gamma parameter = -1, a fixed dynamic version of gamma is used:
# gamma = num_no_misclassified_pixels/num_total_pixels
#---------------------------------------------------------------------
def targeted_patch_composition(input, target, patch_mask, weight=None, size_average=True, gamma = 0.8):
    
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
    



#---------------------------------------------------------------------
# Multi-input of the untargeted_patch_composition loss function (to consider also aux_logits)
#---------------------------------------------------------------------
def multi_scale_patch_composition_targeted(input, target, weight=None, patch_mask = None, size_average=True, scale_weight=None, gamma = 0.9):
    if not isinstance(input, tuple):
        return targeted_patch_composition(input=input, target=target, patch_mask = patch_mask, weight=weight, size_average=size_average, gamma=gamma)

    # Auxiliary weight
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 1.0 # > 1.0 means give more impotance to scaled outputs
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(input[0].device).detach()

    loss_no_misc, loss_misc, ret_gamma = 0, 0, None

    if not isinstance(target, tuple):
        target = [target.clone()] * len(input)

    for i,_ in enumerate(input):
        out_loss_no_misc, out_loss_misc, out_gamma  = targeted_patch_composition(
            input=input[i], target=target[i], weight=weight, patch_mask = patch_mask, size_average=size_average, gamma=gamma)
        
        loss_no_misc  +=  scale_weight[i] * out_loss_no_misc
        loss_misc     +=  scale_weight[i] * out_loss_misc
        ret_gamma      =   out_gamma if (ret_gamma is None) else ret_gamma

    return loss_no_misc, loss_misc, ret_gamma

