import torch
import torch.nn as nn

import numpy as np
import random

import mmcv

from .base import BaseAttacker
from .builder import ATTACKER
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.models.builder import LOSSES


@ATTACKER.register_module()
class PGD(BaseAttacker):
    def __init__(self,
                 epsilon,
                 step_size,
                 num_steps,
                 loss_fn,
                 assigner,
                 category="Madry",
                 rand_init=False,
                 single_camera=False,
                 mono_model=False,
                 *args, 
                 **kwargs):
        """ PGD pixel attack
        Args:
            epsilon (float): L_infty norm bound for visual percep
            step_size (float): step size of one attack iteration
            num_steps (int): attack iteration number
            loss_fn (class): adversarial objective function
            category (str): `trades` or `Madry`, which type of initialization of attack
            rand_init (bool): random initialize adversarial noise or zero initialize
            assigner (class): assign prediction bbox to ground truth bbox
            single_camera (bool): only attack random choose single camera
        """
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.assigner = BBOX_ASSIGNERS.build(assigner)
        self.loss_fn = LOSSES.build(loss_fn)
        self.category = category
        self.single_camera = single_camera
        self.mono_model = mono_model
        self.rand_init = rand_init

        if self.mono_model:
            self.size = (1, 3, 1, 1) # do not have stereo camera information
        else:
            self.size = (1, 1, 3, 1, 1)

        # when attack mono model, can only attack one camera only
        if mono_model:
            assert not single_camera, \
                f"When attack mono detetoc, single_camera should be set to False, but now {single_camera}"

        if isinstance(epsilon, list) or isinstance(epsilon, tuple):
            self.epsilon = torch.tensor(epsilon).view(self.size)
        if isinstance(step_size, list) or isinstance(step_size, tuple):
            self.step_size = torch.tensor(step_size).view(self.size)

    def run(self, model, img_inputs, img_metas, gt_bboxes_3d, gt_labels_3d):
        """Run PGD attack optimization
        Args:
            model (nn.Module): model to be attacked
            img (DataContainer): [B, M, C, H, W]
            img_metas (DataContainer): img_meta information
            gt_bboxes_3d: ground truth of bboxes
            gt_labels_3d: ground truth of labels
        Return:
            inputs: (dict) {'img': img, 'img_metas': img_metas}
        """
        model.eval()

        # img_ = img_inputs[0].data[0].clone()
        # custom for attack BEVDepth
        img_ = img_inputs[0][0].clone()
        B = img_.size(0)
        assert B == 1, f"Batchsize should set to 1 in attack, but now is {B}"
        # only calculate grad of single camera image
        if self.single_camera:
            B, M, C, H, W = img_.size()
            camera = random.randint(0, 5)
            camera_mask = torch.zeros((B, M, C, H, W))
            camera_mask[:, camera] = 1

        if self.category == "trades":
            if self.single_camera:
                x_adv = img_.detach() + camera_mask * self.epsilon * torch.randn(img_.shape).to(img_.device).detach() if self.rand_init else img_.detach()
            else:
                x_adv = img_.detach() + self.epsilon * torch.randn(img_.shape).to(img_.device).detach() if self.rand_init else img_.detach()

        if self.category == "Madry":
            if self.single_camera:
                x_adv = img_.detach() + camera_mask * torch.from_numpy(np.random.uniform(-self.epsilon, self.epsilon, img_.shape)).float().to(img_.device) if self.rand_init else img_.detach()
            else:
                x_adv = img_.detach() + torch.from_numpy(np.random.uniform(-self.epsilon, self.epsilon, img_.shape)).float().to(img_.device) if self.rand_init else img_.detach()

        x_adv = torch.clamp(x_adv, self.lower.view(self.size), self.upper.view(self.size))

        for k in range(self.num_steps):
        
            x_adv.requires_grad_()
            img_inputs[0][0] = x_adv
            inputs = {'img_inputs': img_inputs, 'img_metas': img_metas}
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
            if self.single_camera:
                eta = eta * camera_mask
            x_adv = x_adv.detach() + eta
            x_adv = torch.min(torch.max(x_adv, img_ - self.epsilon), img_ + self.epsilon)
            x_adv = torch.clamp(x_adv, self.lower.view(self.size), self.upper.view(self.size))


        img_inputs[0][0] = x_adv.detach()
        torch.cuda.empty_cache()

        return {'img_inputs': img_inputs, 'img_metas':img_metas}

