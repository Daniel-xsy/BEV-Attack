import torch
import random

from .base import BaseAttacker
from .builder import ATTACKER
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.models.builder import LOSSES


@ATTACKER.register_module()
class FGSM(BaseAttacker):
    def __init__(self,
                 epsilon,
                 loss_fn,
                 assigner,
                 single_camera=False,
                 mono_model=False,
                 *args, 
                 **kwargs):
        """ FGSM pixel attack
        Args:
            epsilon (float): L_infty norm bound for visual perception
            loss_fn (class): adversarial objective function
            assigner (class): assign prediction bbox to ground truth bbox
            single_camera (bool): only attack random choose single camera
        """
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.assigner = BBOX_ASSIGNERS.build(assigner)
        self.loss_fn = LOSSES.build(loss_fn)
        self.single_camera = single_camera
        self.mono_model = mono_model

        if self.mono_model:
            self.size = (1, 3, 1, 1)  # do not have stereo camera information
        else:
            self.size = (1, 1, 3, 1, 1)

        if isinstance(epsilon, list or tuple):
            self.epsilon = torch.tensor(epsilon).view(self.size)

    def run(self, model, img, img_metas, gt_bboxes_3d, gt_labels_3d):
        """Run FGSM attack optimization
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

        camera = random.randint(0, 5)

        img_ = img[0].data[0].clone()
        B = img_.size(0)
        assert B == 1, f"Batchsize should set to 1 in attack, but now is {B}"
        # only calculate grad of single camera image
        if self.single_camera:
            B, M, C, H, W = img_.size()
            camera_mask = torch.zeros((B, M, C, H, W))
            camera_mask[:, camera] = 1

        x_adv = img_.detach()
        x_adv.requires_grad_()

        img[0].data[0] = x_adv
        inputs = {'img': img, 'img_metas': img_metas}

        try:
            outputs = model(return_loss=False, rescale=True, adv_mode=True, **inputs)
        except:
            outputs = model(return_loss=False, rescale=True, **inputs)  # adv_mode=True, 

        assign_results = self.assigner.assign(outputs, gt_bboxes_3d, gt_labels_3d)
        if assign_results is None:
            return {'img': img, 'img_metas':img_metas}

        loss_adv = self.loss_fn(**assign_results)
        loss_adv.backward()

        eta = self.epsilon * x_adv.grad.sign()
        if self.single_camera:
            eta = eta * camera_mask

        x_adv = x_adv.detach() + eta
        x_adv = torch.clamp(x_adv, img_ - self.epsilon, img_ + self.epsilon)
        x_adv = torch.clamp(x_adv, self.lower.view(self.size), self.upper.view(self.size))

        img[0].data[0] = x_adv.detach()

        torch.cuda.empty_cache()
        return {'img': img, 'img_metas':img_metas}