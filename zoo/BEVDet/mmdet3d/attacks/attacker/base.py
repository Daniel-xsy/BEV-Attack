import torch
import torch.nn as nn

import numpy as np

from nuscenes import NuScenes

from .builder import ATTACKER


@ATTACKER.register_module()
class BaseAttacker:
    def __init__(self, img_norm, totensor=False):
        self.img_norm = img_norm
        self.totensor = totensor
        self.upper, self.lower = self._get_bound()

    def run(self, model, data):
        pass

    def _get_bound(self):
        """Calculate max/min pixel value bound
        """

        if self.totensor:
            max_ = 1.0
            min_ = 0.0
        else:
            max_ = 255.0
            min_ = 0.0

        upper = (max_ - torch.tensor(self.img_norm['mean'])) / torch.tensor(self.img_norm['std'])
        lower = min_ - torch.tensor(self.img_norm['mean']) / torch.tensor(self.img_norm['std'])
        
        return upper, lower
