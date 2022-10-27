import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class ClassficationObjective(nn.Module):
    def __init__(self, activate=False):
        """Classification adversarial objective, try to modified the classification result
        of detection model

        Args:
             activate: whether the input is logits and probability distribution
        """
        super().__init__()
        if activate == True:
            self.loss = nn.NLLLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward(self, pred_logits, gt_label, pred_bboxes=None, pred_scores=None, gt_bbox=None):
        """Adversarial Loss

        Args:
            pred_logits (torch.Tensor): [N, C]
            gt_label (torch.Tensor): [N, 1]

        Return:
            loss (torch.Tensor)
        """
        cls_loss = self.loss(pred_logits, gt_label)
        return cls_loss


@LOSSES.register_module()
class TargetedClassificationObjective(nn.Module):

    # Fix targeted attack for fair comparasion
    # last dimension only used in FCOS3D for background
    TARGETS = torch.tensor((4, 6, 0, 7, 8, 6, 2, 0, 1, 2))

    def __init__(self, num_cls=10, random=True, thresh=0.1):
        """Classification adversarial objective, use targeted adversarial attacks

        Args:
             num_cls (int): number of class
             random (bool): random atatck correct label l_n to l'_n
             thresh (float): threshold in C&W attacks
        """
        super().__init__()

        self.random = random
        self.num_cls = num_cls
        self.thresh = thresh
        # self.targets = self._random_target()

        # TODO: Add ohther attack methods
        assert random, "Only support random targeted attack"

    def _random_target(self):
        """Initiate target mapx
        """
        targets = []
        labels = np.array(np.arange(self.num_cls))
        for i in range(self.num_cls):
            target = np.random.choice(np.delete(labels, i))
            targets.append(target)
        
        return torch.LongTensor(targets)

    def _map(self, gt_labels):
        """Map ground truth label to target label
        """
        target_labels = self.TARGETS[gt_labels.squeeze()]
        return target_labels

    def cw_loss(self, correct_score, target_score, thresh=0.1):
        """C&W Attack loss function
        
        correct_score (torch.Tensor): [N, 1]
        target_score (torch.Tensor): [N, 1]
        """

        loss = F.relu(correct_score + thresh - target_score)
        return -1 * loss.mean()

    def forward(self, pred_logits, gt_label, pred_bboxes=None, pred_scores=None, gt_bbox=None):
        """Adversarial Loss

        Args:
            pred_logits (torch.Tensor): [N, C]
            gt_label (torch.Tensor): [N, 1]

        Return:
            loss (torch.Tensor)
        """
        if self.random:
            target_label = self._map(gt_label)

            target_label = target_label.view(1, -1)
            gt_label = gt_label.view(1, -1)

            target_score = torch.gather(pred_logits, dim=-1, index=target_label)
            correct_score = torch.gather(pred_logits, dim=-1, index=gt_label)

            loss = self.cw_loss(correct_score, target_score, self.thresh)

        else:
            assert False, "Only support random targeted attack"
        return loss


@LOSSES.register_module()
class LocalizationObjective(nn.Module):
    def __init__(self,
                 l2loss=False,
                 loc=True,
                 vel=False,
                 orie=False):
        """Localization adversarial objective, try to confuse the localization result
        of detection model
        """
        super().__init__()
        if l2loss:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.L1Loss()
        
        self.loc = loc
        self.vel = vel
        self.orie = orie
        assert loc or vel or orie, "At least specify one target to attack"

    def forward(self, pred_bboxes, gt_bbox, pred_scores=None, pred_logits=None, gt_label=None):
        """Adversarial Loss

        Args:
            pred_bboxes (torch.Tensor): [N, 9] [x, y, z, x_size, y_size, z_size, orientation, vel_x, vel_y]
            gt_bbox (torch.Tensor): [N, 9]

        Return:
            loss (torch.Tensor)
        """
        loc_pred = pred_bboxes[:, :6]
        loc_gt = gt_bbox[:, :6]

        orie_pred = pred_bboxes[:, 6:7]
        orie_gt = gt_bbox[:, 6:7]

        vel_pred = pred_bboxes[:, 7:]
        vel_gt = gt_bbox[:, 7:]

        loss = 0
        if self.loc:
            loss += self.loss(loc_pred, loc_gt)
        if self.orie:
            loss += self.loss(orie_pred, orie_gt)
        if self.vel:
            loss += self.loss(vel_pred, vel_gt)

        return loss