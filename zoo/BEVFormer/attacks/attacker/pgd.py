import torch
import torch.nn as nn

import numpy as np

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
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.loss_fn = loss_fn
        self.category = category
        self.rand_init = rand_init

    def run(self, model, img, img_metas, gt_bboxes_3d, gt_labels_3d):
        model.eval()

        img_ = img[0].data[0].clone()

        if self.category == "trades":
            x_adv = img_.detach() + 0.001 * torch.randn(img_.shape).to(img_.device).detach() if self.rand_init else img_.detach()

        if self.category == "Madry":
            x_adv = img_.detach() + torch.from_numpy(np.random.uniform(-self.epsilon, self.epsilon, img_.shape)).float().to(img_.device) if self.rand_init else img_.detach()
            # x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for k in range(self.num_steps):
        
            x_adv.requires_grad_()
            img[0].data[0] = x_adv
            inputs = {'img': img, 'img_metas': img_metas}
            with torch.no_grad():
                outputs = model(return_loss=False, rescale=True, **inputs)
            mmcv.dump({
                'outputs': outputs,
                'gt_bboxes_3d': gt_bboxes_3d,
                'gt_labels_3d': gt_labels_3d
            }, 'test.pkl')
            model.zero_grad()

            with torch.enable_grad():
                # -----------------------------
                # TODO: add GT in attacker
                # -----------------------------
                loss_adv = self.loss_fn(outputs, gt_bboxes_3d, gt_labels_3d)

            loss_adv.backward()
            eta = self.step_size * x_adv.grad.sign()
            x_adv = x_adv.detach() + eta
            x_adv = torch.min(torch.max(x_adv, img_ - self.epsilon), img_ + self.epsilon)
            # x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv

