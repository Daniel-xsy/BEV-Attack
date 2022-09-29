import torch
import torch.nn as nn

import numpy as np
import random

import mmcv

from attacks.attacker.base import BaseAttacker
from attacks.attacker.builder import ATTACKER


@ATTACKER.register_module()
class PGD(BaseAttacker):
    """PGD Attack
    """
    def __init__(self,
                 epsilon,
                 step_size,
                 num_steps,
                 loss_fn,
                 category,
                 rand_init,
                 assigner,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.loss_fn = loss_fn
        self.category = category
        self.rand_init = rand_init
        self.assigner = assigner

    def run(self, model, img, img_metas, gt_bboxes_3d, gt_labels_3d):
        model.eval()

        camera = random.randint(0, 5)

        img_ = img[0].data[0].clone()
        B, M, C, H, W = img_.size()
        # only calculate grad of single camera image
        camera_mask = torch.zeros((B, M, C, H, W))
        camera_mask[:, camera] = 1

        if self.category == "trades":
            x_adv = img_.detach() + 0.001 * torch.randn(img_.shape).to(img_.device).detach() if self.rand_init else img_.detach()

        if self.category == "Madry":
            x_adv = img_.detach() + torch.from_numpy(np.random.uniform(-self.epsilon, self.epsilon, img_.shape)).float().to(img_.device) if self.rand_init else img_.detach()
        x_adv = x_adv * camera_mask
        x_adv = torch.clamp(x_adv, self.lower.view(1, 1, C, 1, 1), self.upper.view(1, 1, C, 1, 1))

        for k in range(self.num_steps):
        
            x_adv.requires_grad_()
            img[0].data[0] = x_adv
            inputs = {'img': img, 'img_metas': img_metas}
            # with torch.no_grad():
            outputs = model(return_loss=False, rescale=True, **inputs)
            # assign pred bbox to ground truth
            assign_results = self.assigner.assign(outputs, gt_bboxes_3d, gt_labels_3d)
            # no prediction are assign to ground truth, stop attack
            if assign_results is None:
                break
            loss_adv = self.loss_fn(**assign_results)

            loss_adv.backward()
            eta = self.step_size * x_adv.grad.sign()
            eta = eta * camera_mask
            x_adv = x_adv.detach() + eta
            x_adv = torch.min(torch.max(x_adv, img_ - self.epsilon), img_ + self.epsilon)
            x_adv = torch.clamp(x_adv, self.lower.view(1, 1, C, 1, 1), self.upper.view(1, 1, C, 1, 1))


        img[0].data[0] = x_adv.detach()
        torch.cuda.empty_cache()

        return {'img': img, 'img_metas':img_metas}

