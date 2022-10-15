import torch
import torch.nn as nn

import numpy as np

from nuscenes import NuScenes

from .builder import ATTACKER


@ATTACKER.register_module()
class BaseAttacker:
    def __init__(self, img_norm):
        self.img_norm = img_norm
        self.upper, self.lower = self._get_bound()

    def run(self, model, data):
        pass

    def _get_bound(self):
        """Calculate max/min pixel value bound
        """
        upper = (255.0 - torch.tensor(self.img_norm['mean'])) / torch.tensor(self.img_norm['std'])
        lower = - torch.tensor(self.img_norm['mean']) / torch.tensor(self.img_norm['std'])
        
        return upper, lower
