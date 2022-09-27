import torch
import torch.nn as nn

import numpy as np

from nuscenes import NuScenes

from attacks.attacker.builder import ATTACKER


@ATTACKER.register_module()
class BaseAttacker:
    def __init__(self):
        pass

        # self.version = version
        # self.data_root = data_root

        # eval_set_map = {
        #     'v1.0-mini': 'mini_val',
        #     'v1.0-trainval': 'val',
        # }

        # # load ground truth to attack
        # self.nusc = NuScenes(version=self.version, dataroot=self.data_root,
        #                      verbose=True)
        
        # self.nusc = 1
        # self.gt_boxes = load_gt(self.nusc, eval_set_map[self.version], DetectionBox_modified, verbose=True)


    def run(self, model, data):
        pass

