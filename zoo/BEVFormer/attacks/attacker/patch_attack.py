import torch
import torch.nn as nn

import numpy as np

import mmcv

from attacks.attacker.base import BaseAttacker
from attacks.attacker.builder import ATTACKER


@ATTACKER.register_module()
class PatchAttack(BaseAttacker):
    """PGD Attack
    """
    def __init__(self,
                 loss_fn,
                 assigner,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.loss_fn = loss_fn
        self.assigner = assigner

    def run(self, model, img, img_metas, gt_bboxes_3d, gt_labels_3d):
        
        pass

