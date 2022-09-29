'''
Giulio Rossolini
Eval script
'''

import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import time

from task_interfaces import get_task_interface
from attacks.PatchOpt import PatchOPT
import patch_utils

from torch.utils import data
import torch.nn as nn


def optimize_patches(cfg):
    # Setup gpu mode
    device = torch.device("cuda")
    torch.cuda.set_device(cfg["device"]["gpu"])

    # Setup seeds
    torch.manual_seed(cfg.get("seed", cfg["seed"]))
    torch.cuda.manual_seed(cfg.get("seed", cfg["seed"]))
    np.random.seed(cfg.get("seed", cfg["seed"]))
    random.seed(cfg.get("seed", cfg["seed"]))

    # Setup task_interface
    task_interface_obj = get_task_interface(cfg['task'], cfg)
    task_interface_obj.init_loader()
    task_interface_obj.init_exp_folder()

    # Setup experiment folders
    patch_utils.create_experiment_folders(cfg)

    # resume or generate patches 
    num_patches = cfg["adv_patch"]["num_patches"]
    resume_path = cfg["adv_patch"]["path"]["resume"]
    if resume_path is not None:
        print(os.path.basename(resume_path)) 
        print("Resuming optimization from %s" % resume_path)
        seed_patches = patch_utils.get_N_patches_from_img(resume_path, set_loader=task_interface_obj.train_loader, N=num_patches)
    else:
        seed_patches = patch_utils.get_N_random_patches(cfg["adv_patch"]['attr'], set_loader=task_interface_obj.train_loader, N=num_patches, resize=False)

    # add patches to model
    task_interface_obj.add_patches_to_model(seed_patches)

    task_interface_obj.model.eval() 
    task_interface_obj.model.to(task_interface_obj.device)
    task_interface_obj.evaluate_patches()


    print("-------------End of the optimization----------------")





if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/icnet_patch.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    optimize_patches(cfg)


