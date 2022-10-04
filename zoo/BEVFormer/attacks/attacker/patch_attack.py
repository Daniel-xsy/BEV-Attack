import torch
import torch.nn as nn

from copy import deepcopy
import numpy as np

import mmcv

from attacks.attacker.base import BaseAttacker
from attacks.attacker.builder import ATTACKER


@ATTACKER.register_module()
class PatchAttack(BaseAttacker):
    """PatchAttack
    """
    def __init__(self,
                 step_size,
                 num_steps,
                 loss_fn,
                 assigner,
                 patch_size=None,
                 dynamic_patch_size=False,
                 scale=0.5,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.step_size = step_size
        self.num_steps = num_steps
        self.dynamic_patch = dynamic_patch_size
        self.scale = scale
        self.loss_fn = loss_fn
        self.assigner = assigner
        if patch_size is not None:
            self.patch_size = torch.tensor(patch_size)
        assert patch_size is None or not dynamic_patch_size, \
            "Patch size and dynamic patch size should only activate one, now all activate"
        assert patch_size is not None or dynamic_patch_size, \
            "Should activate one of patch_size and dynamic_patch_size, now all off"

    def run(self, model, img, img_metas, gt_bboxes_3d, gt_labels_3d):
        """Run patch attack optimization
        Args:
            model (nn.Module): model to be attacked
            img (DataContainer): [B, M, C, H, W]
            img_metas (DataContainer): img_meta information
            gt_bboxes_3d: ground truth of bboxes
            gt_labels_3d: ground truth of labels
        Return:
            inputs (dict): {'img': img, 'img_metas': img_metas}
        """
        model.eval()

        img = deepcopy(img)
        img_ = img[0].data[0].clone()
        B, M, C, H, W = img_.size()
        gt_bboxes_3d_ = gt_bboxes_3d[0].data[0][0].clone()
        # project from world coordinate to image coordinate
        center = gt_bboxes_3d_.gravity_center
        corners = gt_bboxes_3d_.corners
        center = torch.cat(
            (center, torch.ones_like(center[..., :1])), -1).unsqueeze(dim=-1)

        lidar2img = img_metas[0].data[0][0]['lidar2img']
        lidar2img = np.asarray(lidar2img)
        lidar2img = center.new_tensor(lidar2img).view(-1, 1, 4, 4)  # (M, 1, 4, 4)
        
        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            center.to(torch.float32)).squeeze(-1)

        # filter out invalid project: object center can be seen only by subset of camera
        eps = 1e-5
        bev_mask = (reference_points_cam[..., 2:3] > eps)

        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_metas[0].data[0][0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0].data[0][0]['img_shape'][0][0]
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))

        # valid image plane positions
        reference_points_cam = (reference_points_cam * bev_mask * torch.tensor((W, H))).int()
        patch_mask = torch.zeros_like(img_)

        # get patch mask of original image
        if self.dynamic_patch:
            patch_size = self.get_patch_size(corners, lidar2img, bev_mask, scale=self.scale)
        patch_mask = self.get_patch_mask(reference_points_cam, bev_mask, patch_mask, \
                                                patch_size if self.dynamic_patch else self.patch_size)

        # use pixel mean to random init patch
        x_adv = torch.tensor(self.img_norm['mean']).view(1,1,3,1,1) * torch.randn(img_.shape).to(img_.device).detach()
        # x_adv = torch.ones_like(img_) * 255. # this line is for visualization purpose

        x_adv = x_adv * patch_mask + img_ * (1 - patch_mask)
        x_adv = torch.clamp(x_adv, self.lower.view(1, 1, C, 1, 1), self.upper.view(1, 1, C, 1, 1))

        # optimization
        for k in range(self.num_steps):
        
            x_adv.requires_grad_()
            img[0].data[0] = x_adv
            inputs = {'img': img, 'img_metas': img_metas}
            # with torch.no_grad():
            outputs = model(return_loss=False, rescale=True, adv_mode=True, **inputs)
            # assign pred bbox to ground truth
            assign_results = self.assigner.assign(outputs, gt_bboxes_3d, gt_labels_3d)
            # no prediction are assign to ground truth, stop attack
            if assign_results is None:
                break
            loss_adv = self.loss_fn(**assign_results)

            loss_adv.backward()
            eta = self.step_size * x_adv.grad.sign()
            eta = eta * patch_mask
            x_adv = x_adv.detach() + eta
            x_adv = torch.clamp(x_adv, self.lower.view(1, 1, C, 1, 1), self.upper.view(1, 1, C, 1, 1))


        img[0].data[0] = x_adv.detach()
        torch.cuda.empty_cache()

        return {'img': img, 'img_metas':img_metas}

    def get_patch_mask(self, reference_points_cam, bev_mask, patch_mask, patch_size=torch.tensor((5,5))):
        """Calculate patch mask position for placing patches
        Args:
            reference_points_cam (torch.Tensor): [M, N, 2], M-camera number, N-ground truth, 2-(x, y) position
            bev_mask (torch.Tensor): [M, N, 1], True if the ground truth `n`'s center hit camera `m`, else: False
            mask (torch.Tensor): [B, M, C, H, W]: initial mask, where all the position is 0
            patch_size (torch.Tensor): patch size of each object
        Return:
            mask (torch.Tensor): [B, M, C, H, W]: patch mask, set to 1 if patch exist
        """

        B, M, C, H, W = patch_mask.size()
        M_, N = reference_points_cam.size()[:2]
        assert M == M_, f"camera number in image(f{M}) not equal to camera number in anno(f{M_})"
        assert B == 1, f"Batchsize should be set to 1 when attack, now f{B}"
        # assert (patch_size % 2).any() == 1, f"Patch size should set to odd number, now f{patch_size}"
        assert patch_size.size(-1) == 2, f"Last dim of patch size should have size of 2, now f{patch_size.size(0)}"

        if not self.dynamic_patch:
            patch_size = patch_size.view(1, 1, 2).repeat(M, N, 1)

        # patch size on single side
        patch_size = (patch_size // 2).int()
        bev_mask = bev_mask.squeeze()
        neg_x = torch.maximum(reference_points_cam[..., 0] - patch_size[..., 0], torch.zeros_like(reference_points_cam[..., 0])) * bev_mask
        pos_x = torch.minimum(reference_points_cam[..., 0] + patch_size[..., 0] + 1, W * torch.ones_like(reference_points_cam[..., 0])) * bev_mask
        neg_y = torch.maximum(reference_points_cam[..., 1] - patch_size[..., 1], torch.zeros_like(reference_points_cam[..., 1])) * bev_mask
        pos_y = torch.minimum(reference_points_cam[..., 1] + patch_size[..., 1] + 1, H * torch.ones_like(reference_points_cam[..., 1])) * bev_mask

        for m in range(M):
            for n in range(N):
                # reference point do not hit the image
                if neg_x[m, n] == pos_x[m, n]:
                    continue
                patch_mask[0, m, :, neg_y[m, n] : pos_y[m, n], neg_x[m, n] : pos_x[m, n]] = 1

        return patch_mask

    def get_patch_size(self, corners, lidar2img, bev_mask, scale=0.5):
        """Calculate patch size according to object size on projected image
        Args:
            corner (torch.Tensor): [N, 8, 3], world coordinate corners of 3D bbox
            lidar2img (torch.Tensor): [M, 1, 4, 4], world coordinate to image coordinate matrix
            scale: coefficient patch size to image bbox size
            bev_mask: [M, N, 1], True if ground truth center `n` hit camera `m`

        Output:
            patch_size (torch.Tensor): [M, N, 2], last dim is patch size (h, w) for each ground truth
        """

        N, P = corners.size()[:2]
        M = lidar2img.size(0)
        assert P == 8, f"bbox corners should have 8 points, but now {P}"

        # project corner points to image plane
        corners = torch.cat(
            (corners, torch.ones_like(corners[..., :1])), -1).unsqueeze(dim=-1) # [N, 8, 3] => [N, 8, 4, 1]
        corners = corners.view(N*8, 4, 1)                                       # [N, 8, 4, 1] => [8*N, 4, 1]
        img_corners = torch.matmul(lidar2img.to(torch.float32),                 # [M, 8*N, 4]
                                            corners.to(torch.float32)).squeeze(-1)

        # normalize to image coordinate
        img_corners = img_corners.view(M, N, 8, 4)
        eps = 1e-5
        img_corners = img_corners[..., 0:2] / torch.maximum(
            img_corners[..., 2:3], torch.ones_like(img_corners[..., 2:3]) * eps)

        img_corners = img_corners * bev_mask.view(M, N, 1, 1)      # [M, N, 8, 2], last dim (w, h)
        xmax = img_corners[..., 0].max(dim=-1)[0]
        xmin = img_corners[..., 0].min(dim=-1)[0]
        ymax = img_corners[..., 1].max(dim=-1)[0]
        ymin = img_corners[..., 1].min(dim=-1)[0]

        patch_size = torch.zeros((M, N, 2))
        patch_size[..., 0] = (scale * (xmax - xmin)).int() 
        patch_size[..., 1] = (scale * (ymax - ymin)).int()
        
        return patch_size


@ATTACKER.register_module()
class UniversalPatchAttack(PatchAttack):
    """Universal PatchAttack, one fixed patch pattern adversarial to all scenes and object
    TODO: build this one
    """
    def __init__(self,
                 step_size,
                 num_steps,
                 loss_fn,
                 assigner,
                 patch_size,
                 dynamic_patch=False,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.step_size = step_size
        self.num_steps = num_steps
        self.loss_fn = loss_fn
        self.assigner = assigner
        self.patch_size = torch.tensor(patch_size)

    def run(self, model, img, img_metas, gt_bboxes_3d, gt_labels_3d):
        """Run patch attack optimization
        Args:
            model (nn.Module): model to be attacked
            img (DataContainer): [B, M, C, H, W]
            img_metas (DataContainer): img_meta information
            gt_bboxes_3d: ground truth of bboxes
            gt_labels_3d: ground truth of labels
        Return:
            inputs (dict): {'img': img, 'img_metas': img_metas}
        """
        model.eval()

        img = deepcopy(img)
        img_ = img[0].data[0].clone()
        B, M, C, H, W = img_.size()
        gt_bboxes_3d_ = gt_bboxes_3d[0].data[0][0].clone()
        # project from world coordinate to image coordinate
        center = gt_bboxes_3d_.gravity_center
        center = torch.cat(
            (center, torch.ones_like(center[..., :1])), -1).unsqueeze(dim=-1)

        lidar2img = img_metas[0].data[0][0]['lidar2img']
        lidar2img = np.asarray(lidar2img)
        lidar2img = center.new_tensor(lidar2img).view(-1, 1, 4, 4)  # (M, 1, 4, 4)
        
        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            center.to(torch.float32)).squeeze(-1)

        # filter out invalid project: object center can be seen only by subset of camera
        eps = 1e-5
        bev_mask = (reference_points_cam[..., 2:3] > eps)

        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_metas[0].data[0][0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0].data[0][0]['img_shape'][0][0]
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))

        # valid image plane positions
        reference_points_cam = (reference_points_cam * bev_mask * torch.tensor((W, H))).int()
        patch_mask = torch.zeros_like(img_)

        # get patch mask of original image
        patch_mask = self.get_patch_mask(reference_points_cam, bev_mask, patch_mask, self.patch_size)

        # use pixel mean to random init patch
        x_adv = torch.tensor(self.img_norm['mean']).view(1,1,3,1,1) * torch.randn(img_.shape).to(img_.device).detach()

        x_adv = x_adv * patch_mask + img_ * (1 - patch_mask)
        x_adv = torch.clamp(x_adv, self.lower.view(1, 1, C, 1, 1), self.upper.view(1, 1, C, 1, 1))

        # optimization
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
            eta = eta * patch_mask
            x_adv = x_adv.detach() + eta
            x_adv = torch.clamp(x_adv, self.lower.view(1, 1, C, 1, 1), self.upper.view(1, 1, C, 1, 1))


        img[0].data[0] = x_adv.detach()
        torch.cuda.empty_cache()

        return {'img': img, 'img_metas':img_metas}

    def get_patch_mask(self, reference_points_cam, bev_mask, patch_mask, patch_size=torch.tensor((5,5))):
        """Calculate patch mask position for placing patches
        Args:
            reference_points_cam (torch.Tensor): [M, N, 2], M-camera number, N-ground truth, 2-(x, y) position
            bev_mask (torch.Tensor): [M, N, 1], True if the ground truth `n`'s center hit camera `m`, else: False
            mask (torch.Tensor): [B, M, C, H, W]: initial mask, where all the position is 0
            patch_size (torch.Tensor): patch size of each object
        Return:
            mask (torch.Tensor): [B, M, C, H, W]: patch mask, set to 1 if patch exist
        """

        B, M, C, H, W = patch_mask.size()
        M_, N = reference_points_cam.size()[:2]
        assert M == M_, f"camera number in image(f{M}) not equal to camera number in anno(f{M_})"
        assert B == 1, f"Batchsize should be set to 1 when attack, now f{B}"
        assert (patch_size % 2).any() == 1, f"Patch size should set to odd number, now f{patch_size}"
        assert patch_size.size(0) == 2, f"Patch size should have size of 2, now f{patch_size.size(0)}"

        # patch size on single side
        patch_size = patch_size // 2
        bev_mask = bev_mask.squeeze()
        neg_x = torch.maximum(reference_points_cam[..., 0] - patch_size[0], torch.zeros_like(reference_points_cam[..., 0])) * bev_mask
        pos_x = torch.minimum(reference_points_cam[..., 0] + patch_size[0] + 1, W * torch.ones_like(reference_points_cam[..., 0])) * bev_mask
        neg_y = torch.maximum(reference_points_cam[..., 1] - patch_size[1], torch.zeros_like(reference_points_cam[..., 1])) * bev_mask
        pos_y = torch.minimum(reference_points_cam[..., 1] + patch_size[1] + 1, H * torch.ones_like(reference_points_cam[..., 1])) * bev_mask

        for m in range(M):
            for n in range(N):
                # reference point do not hit the image
                if neg_x[m, n] == pos_x[m, n]:
                    continue
                patch_mask[0, m, :, neg_y[m, n] : pos_y[m, n], neg_x[m, n] : pos_x[m, n]] = 1

        return patch_mask

    def _get_patch_size(self, reference_points_cam, gt_bboxes_3d, bev_mask):
        """Calculate patch size according to object size, use different patch size for different object,
        which is more make sense than fixed size patch.
        Args:
            reference_points_cam (torch.Tensor): [M, N, 2], M-camera number, N-ground truth, 2-(x, y) position
            gt_bboxes_3d (DataContainer): ground truth bboxes
            mask (torch.Tensor): [B, M, C, H, W]: initial mask, where all the position is 0
        """
        pass
