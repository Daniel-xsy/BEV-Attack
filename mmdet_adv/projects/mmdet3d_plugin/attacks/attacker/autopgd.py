import torch
import torch.nn as nn
import math

import numpy as np
import random

import mmcv

from .base import BaseAttacker
from .builder import ATTACKER
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.models.builder import LOSSES


@ATTACKER.register_module()
class AutoPGD(BaseAttacker):
    def __init__(self,
                 epsilon,
                 num_steps,
                 loss_fn,
                 assigner,
                 category="Madry",
                 rand_init=False,
                 single_camera=False,
                 mono_model=False,
                 sequential=False,
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
            sequential (bool): sequential inputs in BEVDet4D
        """
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        if isinstance(epsilon, list) or isinstance(epsilon, tuple):
            self.step_size = [0.2 * eps for eps in epsilon]
        elif isinstance(epsilon, float):
            self.step_size = 0.2 * epsilon
        else:
            raise NotImplementedError
        self.num_steps = num_steps
        self.assigner = BBOX_ASSIGNERS.build(assigner)
        self.loss_fn = LOSSES.build(loss_fn)
        self.category = category
        self.single_camera = single_camera
        self.mono_model = mono_model
        self.rand_init = rand_init
        self.sequential = sequential
        
        self.define_checkpoints()

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
        if isinstance(self.step_size, list) or isinstance(self.step_size, tuple):
            self.step_size = torch.tensor(self.step_size).view(self.size)


    def define_checkpoints(self):
        p = [0, 0.22]
        while p[-1] < 1.0:
            curr_p = p[-1] + max(p[-1] - p[-2] - 0.03, 0.06)
            if curr_p > 1:
                curr_p = 1.0
            p.append(curr_p)
        self.w = [int(pi * self.num_steps) for pi in p]

    def run(self, model, img, img_metas, gt_bboxes_3d, gt_labels_3d):
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

        alpha = 0.75
    
        # img_ = img[0].data[0].clone()
        # custom for attack BEVDepth
        img_ = img[0].data[0].clone()
        B = img_.size(0)
        assert B == 1, f"Batchsize should set to 1 in attack, but now is {B}"
        # only calculate grad of single camera image
        if self.single_camera:
            B, M, C, H, W = img_.size()
            camera = random.randint(0, 5)
            camera_mask = torch.zeros((B, M, C, H, W))
            camera_mask[:, camera] = 1
        # sequential input, only attack current timestamp
        if self.sequential:
            B, M, C, H, W = img_.size()
            assert M % 6 == 0, f"When activate sequential input, camera number must be full divided by 6, now {M}"
            camera_mask = torch.zeros((B, M, C, H, W))
            camera_mask[:, 0 : -1 : 2] = 1

        if self.category == "trades":
            if self.single_camera or self.sequential:
                x_adv = img_.detach() + camera_mask * self.epsilon * torch.randn(img_.shape).to(img_.device).detach() if self.rand_init else img_.detach()
            else:
                x_adv = img_.detach() + self.epsilon * torch.randn(img_.shape).to(img_.device).detach() if self.rand_init else img_.detach()

        if self.category == "Madry":
            if self.single_camera or self.sequential:
                x_adv = img_.detach() + camera_mask * torch.from_numpy(np.random.uniform(-self.epsilon, self.epsilon, img_.shape)).float().to(img_.device) if self.rand_init else img_.detach()
            else:
                x_adv = img_.detach() + torch.from_numpy(np.random.uniform(-self.epsilon, self.epsilon, img_.shape)).float().to(img_.device) if self.rand_init else img_.detach()

        x_adv = torch.clamp(x_adv, self.lower.view(self.size), self.upper.view(self.size))

        if isinstance(self.step_size, float):
            step = self.step_size
        else: 
            step = self.step_size.clone()
        last_half_step = False
        best_loss_increase = True
        best_x_adv = x_adv.clone()
        best_curr_x_adv = x_adv.clone()
        best_adv_loss = 0.0
        no_assign_results = False
        for j in range(1, len(self.w)):
            interval = self.w[j] - self.w[j - 1]
            increase_loss = []
            loss_adv_last = 0.0
            best_current_loss = 0.0
            x_adv_last = x_adv.clone()
            momentum = torch.zeros_like(x_adv)
            for i in range(interval):
                x_adv.requires_grad_()
                img[0].data[0] = x_adv
                inputs = {'img': img, 'img_metas': img_metas}
                # with torch.no_grad():
                outputs = model(return_loss=False, rescale=True, **inputs)
                # assign pred bbox to ground truth
                assign_results = self.assigner.assign(outputs, gt_bboxes_3d, gt_labels_3d)
                # no prediction are assign to ground truth, stop attack
                if assign_results is None:
                    no_assign_results = True
                    break
                loss_adv = self.loss_fn(**assign_results)
                if loss_adv > loss_adv_last:
                    increase_loss.append(True)
                else:
                    increase_loss.append(False)
                loss_adv_last = loss_adv.item()

                if loss_adv.item() > best_current_loss:
                    best_current_loss = loss_adv.item()
                    best_curr_x_adv = x_adv.clone()
                loss_adv.backward()
                eta = step * x_adv.grad.sign()
                if self.single_camera or self.sequential:
                    eta = eta * camera_mask
                    
                z_adv = x_adv.detach() + eta
                z_adv = torch.min(torch.max(z_adv, img_ - self.epsilon), img_ + self.epsilon)
                z_adv = torch.clamp(z_adv, self.lower.view(self.size), self.upper.view(self.size))
                
                x_adv = (x_adv + alpha * (z_adv - x_adv) + (1 - alpha) * momentum).detach()
                x_adv = torch.min(torch.max(x_adv, img_ - self.epsilon), img_ + self.epsilon)
                x_adv = torch.clamp(x_adv, self.lower.view(self.size), self.upper.view(self.size))
                
                momentum = (x_adv - x_adv_last).detach()
                x_adv_last = x_adv.clone()
                torch.cuda.empty_cache()

            if no_assign_results:
                break
            
            if interval == 0:
                continue
            
            if best_current_loss > best_adv_loss:
                best_adv_loss = best_current_loss
                best_x_adv = best_curr_x_adv.clone()
                best_loss_increase = True
            else:
                best_loss_increase = False
                
            # half the step size and restart from the best point
            if sum(increase_loss) / interval < 0.75 or (not last_half_step and not best_loss_increase):
                step = step / 2
                last_half_step = True
            else:
                last_half_step = False
            x_adv = best_x_adv.detach()

        img[0].data[0] = x_adv.detach()
        torch.cuda.empty_cache()

        return {'img': img, 'img_metas':img_metas}

