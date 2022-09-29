'''
Giulio Rossolini
Object Detection Attacks
'''

import torch
import random

from .attacks import Attack
import patch_utils as patch_utils


def norm(v):
    n = torch.norm(v, p=float('2'))
    return (v/n) if n > 0 else v # to avoid NaN error 

def sign(v):
    v = torch.sign(v)
    return v



#-----------------------------------------------------------
# Object Detection Attack Untargeted
#-----------------------------------------------------------
class Attack_OD(Attack):
    r"""
    class for untargeted 2D Object Detection Attack.
    """

    def __init__(self, cfg, opt_obj):
        self.initStat()

        self.opt_sign = [-1, -1, -1, -1]
        self.attacked_class, self.target_class = cfg["adv_patch"]['attr']['attacked_label'], cfg["adv_patch"]['attr']['target_label']
        self.adv_loss_weights = cfg['adv_patch']['patch_opt']['loss']['adv_loss']['mult_factor']
        self.shuffled_classes = list(range(1, 91))
        random.shuffle(self.shuffled_classes)

        self.bbox_target = cfg["adv_patch"]['attr']['bbox']
        
        super().__init__(cfg, opt_obj) 

    #------------------------------
    #Init attack stats 
    def initStat(self):
        self.epoch_loss_info_cls        = 0
        self.epoch_loss_info_bbox       = 0
        self.epoch_loss_info_obj        = 0
        self.epoch_loss_info_rpn_bbox   = 0
        self.epoch_loss_info_samples    = 0
        self.epoch_loss_info_smoothness = 0
        self.epoch_loss_info_NPS        = 0
        return


    #------------------------------
    # Print attack stats 
    def printStat(self, i, reset_stat = True):
        fmt_str = "Epochs [{:d}/{:d}]  Mean Losses: cls loss {:.4f}, bbox loss {:.4f}, obj loss: {:.4f}, rpn_bbox loss {:.4f}, smooth loss {:.4f}, NPS loss {:.4f},(on {:d} training samples) "
        print_str = fmt_str.format(
                i + 1,
                self.cfg["adv_patch"]['patch_opt']["opt_iters"],
                self.epoch_loss_info_cls/self.epoch_loss_info_samples,
                self.epoch_loss_info_bbox/self.epoch_loss_info_samples,
                self.epoch_loss_info_obj/self.epoch_loss_info_samples,
                self.epoch_loss_info_rpn_bbox/self.epoch_loss_info_samples,
                self.epoch_loss_info_smoothness /self.epoch_loss_info_samples,
                self.epoch_loss_info_NPS /self.epoch_loss_info_samples,
                self.epoch_loss_info_samples
            )
        print(print_str)
        i += 1

        if reset_stat:
            self.initStat()

        return

    #--------------------------
    # Label prepatching
    def label_prepatching(self, images, labels):
        self.labels_classes = torch.tensor([d['category_id'] for d in labels]).to(self.opt_obj.device)
        self.labels_bbox = torch.tensor([d['bbox'] for d in labels]).to(self.opt_obj.device)
        self.labels_segm = torch.zeros((self.labels_bbox.shape[0], *images.shape[-2:]))
        for bbox_idx in range(self.labels_bbox.shape[0]):
            self.labels_segm[bbox_idx, 
                        int(self.labels_bbox[bbox_idx][1]):int(self.labels_bbox[bbox_idx][1]+self.labels_bbox[bbox_idx][3]),
                        int(self.labels_bbox[bbox_idx][0]):int(self.labels_bbox[bbox_idx][0]+self.labels_bbox[bbox_idx][2])] = 1
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

        if (self.labels_bbox.numel()) > 0:
            # CHANGE LABELS and bboxes
            """ """
            if self.target_class is not None:
                if  self.attacked_class is not None:
                    self.labels_classes [self.labels_classes  ==  self.attacked_class] = self.target_class
                else:
                    self.labels_classes [:] = self.target_class
            else:
                if  self.attacked_class is not None:
                    self.labels_classes [self.labels_classes  != self.attacked_class] = self.attacked_class
                
            
            if self.bbox_target is not None:
                bbox_target = patch_utils.patch_mask2bbox(patch_masks)
                add_label = torch.tensor([self.target_class]).to(self.obj_opt.device)
                add_bbox = torch.tensor([bbox_target]).to(self.obj_opt.device).float()
                add_bbox[:, 0] /= self.opt_obj.task_interface.curr_img_resize_factors[0][1]
                add_bbox[:, 2] /= self.opt_obj.task_interface.curr_img_resize_factors[0][1]
                add_bbox[:, 1] /= self.opt_obj.task_interface.curr_img_resize_factors[0][0]
                add_bbox[:, 3] /= self.opt_obj.task_interface.curr_img_resize_factors[0][0]

                labels_classes = add_label
                self.labels_bbox = add_bbox

            self.labels_bbox[:, 2:] += self.labels_bbox[:, :2]
            self.labels_bbox = self.labels_bbox.float()
                            
            self.labels_bbox[:, 0] *= self.opt_obj.task_interface.curr_img_resize_factors[0][1]
            self.labels_bbox[:, 2] *= self.opt_obj.task_interface.curr_img_resize_factors[0][1]
            self.labels_bbox[:, 1] *= self.opt_obj.task_interface.curr_img_resize_factors[0][0]
            self.labels_bbox[:, 3] *= self.opt_obj.task_interface.curr_img_resize_factors[0][0]

            labels_segm = patch_masks[:, 0, :, :]

        self.opt_obj.task_interface.curr_target_dict = [{
                'boxes': self.labels_bbox,
                'labels': self.labels_classes,
                'masks': self.labels_segm
            }]

        return perturbed_images, patch_masks


    #--------------------------
    # Compute adv grad
    def compute_adv_grad(self):
        outputs = self.opt_obj.curr_outputs
        labels = self.opt_obj.curr_labels
        patch_masks = self.opt_obj.curr_patch_masks

        if 'loss_classifier' in outputs:
                loss_cls, loss_bbox, loss_obj, loss_rpn_bbox = outputs['loss_classifier'], outputs['loss_box_reg'], outputs['loss_objectness'], outputs['loss_rpn_box_reg']
        else:
            # print(outputs)
            loss_cls, loss_bbox = outputs['classification'], outputs['bbox_regression']
            loss_rpn_bbox = 0 * loss_cls.clone()
            loss_obj = 0 * loss_bbox.clone()


        self.epoch_loss_info_cls        += loss_cls.item()
        self.epoch_loss_info_bbox       += loss_bbox.item()
        self.epoch_loss_info_obj        += loss_obj.item()
        self.epoch_loss_info_rpn_bbox   += loss_rpn_bbox.item()
        self.epoch_loss_info_samples    += patch_masks.shape[0]


        retain_graph_bool = True
        norm_grad_losses = []
        for _ in range(self.opt_obj.num_patches):
            norm_grad_losses.append([None,None, None, None])

        loss_array = [loss_cls, loss_bbox, loss_obj, loss_rpn_bbox]

        # compute and normalize each loss
        for count, l in enumerate(loss_array):
            self.opt_obj.optimizer.zero_grad()
            l.backward(retain_graph=retain_graph_bool)
            
            for patch_idx, p in enumerate(self.opt_obj.model.patches):
                grad_loss = self.opt_obj.model.patches[patch_idx].grad.data.clone().to(self.opt_obj.device)
                norm_grad_losses[patch_idx][count] = norm(grad_loss)

        adv_grad_patches = []
        for patch_idx, p in enumerate(self.opt_obj.model.patches):
            final_grad_adv = 0
            for w_idx, w in enumerate(self.adv_loss_weights):
                final_grad_adv += w * (self.opt_sign[w_idx] * norm_grad_losses[patch_idx][w_idx]) 
            adv_grad_patches.append(norm(final_grad_adv))
        return adv_grad_patches


