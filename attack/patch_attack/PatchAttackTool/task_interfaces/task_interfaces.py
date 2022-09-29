'''
Giulio Rossolini
Task Interface Base Class
'''

import patch_utils
import importlib
import os

from torch.utils import data
import torch

#--------------------------------------------------
# Base class
#--------------------------------------------------
class TaskInterface(object):
    r"""
    NOTE. Use the base class to implement common function,
      which are not specific for the task architecture
    """
    def __init__(self,cfg):
        self.task_type = cfg['task']
        self.cfg = cfg
        self.test_results = []

        # Setup gpu mode
        self.device = torch.device("cuda")
        return
    
    def get_optimizer(self):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError

    def init_loader(self):
        raise NotImplementedError

    def forward(self,images):
        raise NotImplementedError

    def evaluate_patches(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError 

    def item_preprocessing(self, data_items):
        raise NotImplementedError

    def add_patches_to_model(self, seed_patches):
        patch_utils.init_model_N_patches(model = self.model, 
            mode = "train", 
            N = len(seed_patches), 
            seed_patches = seed_patches)
        return

    def init_exp_folder(self):
        cfg = self.cfg
        self.exp_folder = cfg['adv_patch']['path']["out_dir"]
        self.exp_name = cfg['adv_patch']['path']["exp_name"]
        self.exp_root = os.path.join(self.exp_folder, self.exp_name)
        self.patches_folder = os.path.join(self.exp_root, "patches")
        return

    