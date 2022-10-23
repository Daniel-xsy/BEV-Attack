import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
import numpy as np


from .base import BaseAttacker
from .builder import ATTACKER
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.models.builder import LOSSES


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
                 mono_model=False,
                 scale=0.5,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        """
        Args:
            step_size (float): pixel value update step size in one iteration
            num_steps (int): number of iteration to generate adversarial patch
            loss_fn (class): adversarial objective funtion
            assigner (class): assign prediction to ground truth
            patch_size (list): adversarial patch size, None if using dynamic patch size
            denamic_patch_size (bool): when activate, adjust patch size according to object size
            scale (float): patch size scale of object size, in (0, 1)
        """

        self.step_size = step_size
        self.num_steps = num_steps
        self.dynamic_patch = dynamic_patch_size
        self.mono_model = mono_model
        self.scale = scale
        self.assigner = BBOX_ASSIGNERS.build(assigner)
        self.loss_fn = LOSSES.build(loss_fn)
        if patch_size is not None:
            self.patch_size = torch.tensor(patch_size)
        assert patch_size is None or not dynamic_patch_size, \
            "Patch size and dynamic patch size should only activate one, now all activate"
        assert patch_size is not None or dynamic_patch_size, \
            "Should activate one of patch_size and dynamic_patch_size, now all off"
        assert scale > 0 and scale < 1, f"Scale should be chosen from (0, 1), but now: {scale}"

        if mono_model:
            self.size = (1, 3, 1, 1) # do not have stereo camera information
        else:
            self.size = (1, 1, 3, 1, 1)

        if isinstance(step_size, list or tuple):
            self.step_size = torch.tensor(step_size).view(self.size)

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
        B = img_.size(0)
        assert B == 1, f"When attack models, batchsize should be set to 1, but now {B}"
        C, H, W = img_.size()[-3:]
        gt_bboxes_3d_ = gt_bboxes_3d[0].data[0][0].clone()

        # when evaluate monocular model, some camera do not cantain bbox
        if len(gt_bboxes_3d_.tensor) == 0:
            return {'img': img, 'img_metas':img_metas}
        # project from world coordinate to image coordinate
        
        center = deepcopy(gt_bboxes_3d_.gravity_center)
        corners = deepcopy(gt_bboxes_3d_.corners)

        if self.mono_model:
            # when attack monocular models, the coordinate is camera coordinate
            # we need to transform to lidar coordinate first
            center, corners = self.camera2lidar(center, corners, img_metas)

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

        reference_points_cam[..., 0] /= W
        reference_points_cam[..., 1] /= H
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
        x_adv = torch.tensor(self.img_norm['mean']).view(self.size) * torch.randn(img_.shape).to(img_.device).detach()
        # x_adv = torch.ones_like(img_) * 255. # this line is for visualization purpose

        x_adv = x_adv * patch_mask + img_ * (1 - patch_mask)
        x_adv = torch.clamp(x_adv, self.lower.view(self.size), self.upper.view(self.size))

        # optimization
        for k in range(self.num_steps):
        
            x_adv.requires_grad_()
            img[0].data[0] = x_adv
            inputs = {'img': img, 'img_metas': img_metas}
            # with torch.no_grad():
            outputs = model(return_loss=False, rescale=True, adv_mode=True, **inputs) # adv_mode=True, 
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
            x_adv = torch.clamp(x_adv, self.lower.view(self.size), self.upper.view(self.size))


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

        if self.mono_model:
            B, C, H, W = patch_mask.size()
            M = 1
        else:
            B, M, C, H, W = patch_mask.size()
        M_, N = reference_points_cam.size()[:2]
        assert M == M_, f"camera number in image(f{M}) not equal to camera number in anno(f{M_})"
        assert B == 1, f"Batchsize should be set to 1 when attack, now f{B}"
        # assert (patch_size % 2).any() == 1, f"Patch size should set to odd number, now f{patch_size}"
        assert patch_size.size(-1) == 2, f"Last dim of patch size should have size of 2, now f{patch_size.size(0)}"

        if not self.dynamic_patch:
            patch_size = patch_size.view(1, 1, 2).repeat(M, N, 1)

        # patch size on single side
        patch_size = torch.div(patch_size, 2, rounding_mode='floor')
        bev_mask = bev_mask.squeeze()
        neg_x = (torch.maximum(reference_points_cam[..., 0] - patch_size[..., 0], torch.zeros_like(reference_points_cam[..., 0])) * bev_mask).int()
        pos_x = (torch.minimum(reference_points_cam[..., 0] + patch_size[..., 0] + 1, W * torch.ones_like(reference_points_cam[..., 0])) * bev_mask).int()
        neg_y = (torch.maximum(reference_points_cam[..., 1] - patch_size[..., 1], torch.zeros_like(reference_points_cam[..., 1])) * bev_mask).int()
        pos_y = (torch.minimum(reference_points_cam[..., 1] + patch_size[..., 1] + 1, H * torch.ones_like(reference_points_cam[..., 1])) * bev_mask).int()

        for m in range(M):
            for n in range(N):
                # reference point do not hit the image
                if neg_x[m, n] == pos_x[m, n]:
                    continue
                if self.mono_model:
                    patch_mask[0, :, neg_y[m, n] : pos_y[m, n], neg_x[m, n] : pos_x[m, n]] = 1
                else:
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

    def camera2lidar(self, center, corners, img_metas):
        """Convert camera coordinate to lidar coordinate
        """
        assert 'sensor2lidar_translation' in list(img_metas[0].data[0][0].keys())
        assert 'sensor2lidar_rotation' in list(img_metas[0].data[0][0].keys())

        sensor2lidar_translation = np.array(img_metas[0].data[0][0]['sensor2lidar_translation'])
        sensor2lidar_rotation = np.array(img_metas[0].data[0][0]['sensor2lidar_rotation'])

        center = center @ sensor2lidar_rotation.T + sensor2lidar_translation
        corners = corners @ sensor2lidar_rotation.T + sensor2lidar_translation

        return center, corners


@ATTACKER.register_module()
class UniversalPatchAttack(BaseAttacker):
    """Universal PatchAttack, one fixed patch pattern adversarial to all scenes and object
    """
    def __init__(self,
                 step_size,
                 epoch,
                 loader,
                 loss_fn,
                 assigner,
                 category_specify=True,
                 catagory_num=10,
                 patch_size=None,
                 dynamic_patch_size=False,
                 scale=0.5,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        """
        Args:
            step_size (float): pixel value update step size in one iteration
            num_steps (int): number of iteration to generate adversarial patch
            loader (DataLoader): dataloader to optimize patch
            loss_fn (class): adversarial objective funtion
            assigner (class): assign prediction to ground truth
            catagory_specify(bool): if True, using same patch for each catagory, if False, share patch across catagory
            patch_size (list): adversarial patch size, None if using dynamic patch size
            denamic_patch_size (bool): when activate, adjust patch size according to object size
            scale (float): patch size scale of object size, in (0, 1)
        """

        self.step_size = step_size
        self.epoch = epoch
        self.category_specify = category_specify
        self.catagory_num = catagory_num
        self.dynamic_patch = dynamic_patch_size
        self.scale = scale
        self.loader = loader
        self.assigner = BBOX_ASSIGNERS.build(assigner)
        self.loss_fn = LOSSES.build(loss_fn)
        assert patch_size is not None
        self.patch_size = torch.tensor(patch_size)

        assert scale > 0 and scale < 1, f"Scale should be chosen from (0, 1), but now: {scale}"
        assert not category_specify or catagory_num != 0, f"When catagory specify is activated, catagory number can't be 0"
        # init patch
        self.patches = self._init_patch()

    def _init_patch(self):
        """Initilize patch pattern

        Return:
            patches (torch.Tensor): init patch
        """

        catagory_num = self.catagory_num if self.category_specify else 1

        # init patch pixel uniformly from [0, 255]
        patches = 255 * torch.rand((catagory_num, 3, self.patch_size[0], self.patch_size[1]))
        # patches = 255 * torch.ones((catagory_num, 3, self.patch_size[0], self.patch_size[1])) # only for debug usage
        # normalize
        patches = (patches - torch.tensor(self.img_norm['mean']).view(1, 3, 1, 1)) / torch.tensor(self.img_norm['std']).view(1, 3, 1, 1)

        return patches
        
    def train(self, model):
        """Run patch attack optimization
        Args:
            model (nn.Module): victim model
            loader (Dataset): data loader
        Return:
            patches (torch.Tensor): universal adversarial patches
        """
        model.eval()

        # add momentum to gradient
        eta_prev = 0
        momentum = 0.8
        for i in range(self.epoch):
            for batch_id, data in enumerate(self.loader):

                self.patches.requires_grad_()

                img, img_metas = data['img'], data['img_metas']
                gt_bboxes_3d = data['gt_bboxes_3d']
                gt_labels_3d = data['gt_labels_3d']

                reference_points_cam, bev_mask, patch_size = self.get_reference_points(img, img_metas, gt_bboxes_3d)
                inputs = self.place_patch(img, img_metas, gt_labels_3d, reference_points_cam, bev_mask, patch_size)
                outputs = model(return_loss=False, rescale=True, adv_mode=True, **inputs)
                # assign pred bbox to ground truth
                assign_results = self.assigner.assign(outputs, gt_bboxes_3d, gt_labels_3d)
                # use sgd, multiply -1 to max loss_adv
                loss_adv = self.loss_fn(**assign_results)

                loss_adv.backward()

                eta = momentum * eta_prev + (1 - momentum) * self.step_size * self.patches.grad.sign()
                # update
                eta_prev = eta
                self.patches = self.patches.detach() + eta
                self.patches = torch.clamp(self.patches, self.lower.view(1, 3, 1, 1), self.upper.view(1, 3, 1, 1))
                print(f'[Epoch: {i}/{self.epoch}] Iteration: {batch_id}/{len(self.loader)}  Loss: {loss_adv}')

        return self.patches

    def run(self, model, img, img_metas, gt_bboxes_3d, gt_labels_3d):
        """Paste patch on the image on-the-fly

        Args:
            img (DataContainer): input image data
            img_metas (DataContainer): image meta information
        Return:
            inputs (dict): {'img': img, 'img_metas': img_metas}
        """
        reference_points_cam, bev_mask, patch_size = self.get_reference_points(img, img_metas, gt_bboxes_3d)
        inputs = self.place_patch(img, img_metas, gt_labels_3d, reference_points_cam, bev_mask, patch_size)
        return inputs

    def place_patch(self, img, img_metas, gt_labels, reference_points_cam, bev_mask, patch_size=torch.tensor((5,5))):
            """Place patch to the center of object
            Args:
                img (torch.Tensor): [B, M, C, H, W], which img to add patch
                img_metas (DataContainer): image meta information
                patches (torch.Tensor): [cls_num, 3, H', W'], patches need to add
                patches (torch.Tensor): [cls_num, 3, H', W'], patches need to add
                gt_labels (torch.Tensor): [N, ], Only useful when category_specify is activate, ground truth label to specify which patch to use
                reference_points_cam (torch.Tensor): [M, N, 2], M-camera number, N-ground truth, 2-(x, y) position
                bev_mask (torch.Tensor): [M, N, 1], True if the ground truth `n`'s center hit camera `m`, else: False
                patch_size (torch.Tensor): patch size of each object
            Reurn:
                patch_img (torch.Tensor): pacthed image
            """

            img_ = img[0].data[0].clone()
            gt_labels = gt_labels[0].data[0][0]
            B, M, C, H, W = img_.size()
            M_, N = reference_points_cam.size()[:2]
            assert M == M_, f"camera number in image(f{M}) not equal to camera number in reference_points_cam(f{M_})"
            assert B == 1, f"Batchsize should be set to 1 when attack, now f{B}"
            # assert (patch_size % 2).any() == 1, f"Patch size should set to odd number, now f{patch_size}"
            assert patch_size.size(-1) == 2, f"Last dim of patch size should have size of 2, now f{patch_size.size(0)}"

            # fixed patch size
            if not self.dynamic_patch:
                patch_size = patch_size.view(1, 1, 2).repeat(M, N, 1)

            # use the same patch for all category
            patches = self.patches
            if not self.category_specify:
                assert patches.size(0) == 1, \
                            f"When using same patch to all category, the first dim of patches is expected to be 1, now{patches.size(0)}"
                patches = patches.repeat(10, 1, 1, 1)

            # patch size on single side
            patch_size = torch.div(patch_size, 2, rounding_mode='floor')
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
                    w_, h_ = pos_x[m, n] - neg_x[m, n], pos_y[m, n] - neg_y[m, n]
                    # resize patch size
                    resize_patch = F.interpolate(patches[gt_labels[n]].unsqueeze(dim=0), size=(h_, w_), mode='bilinear', align_corners=True).squeeze()
                    img_[0, m, :, neg_y[m, n] : pos_y[m, n], neg_x[m, n] : pos_x[m, n]] = resize_patch
                    a = 1

            img[0].data[0] = img_
            return {'img': img, 'img_metas': img_metas}

    def get_reference_points(self, img, img_metas, gt_bboxes_3d):

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

        if self.dynamic_patch:
            patch_size = self.get_patch_size(corners, lidar2img, bev_mask, scale=self.scale)
        else:
            patch_size = self.patch_size

        return reference_points_cam, bev_mask, patch_size

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