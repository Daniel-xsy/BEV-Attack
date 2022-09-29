'''
Giulio Rossolini
Patch Optimization Class designed to run the scene-specific attack 
'''

import os
import torch
#from tqdm import tqdm

import patch_utils as patch_utils

def get_attack(cfg, opt_obj):
    attack_name = cfg['adv_patch']['patch_opt']['attack']
    print(attack_name)
    
    if attack_name == 'untargeted_SS':
        from .attacks_SS import Attack_Untargeted_SS
        return Attack_Untargeted_SS(cfg, opt_obj)

    elif attack_name == 'untargeted_depth':
        from .attacks_depth import Attack_Untargeted_depth
        return Attack_Untargeted_depth(cfg, opt_obj)
    
    elif attack_name == 'untargeted_3dod':
        from .attacks_3DOD import Attack_3DOD
        return Attack_3DOD(cfg, opt_obj)
    
    elif attack_name == 'attack_od':
        from .attacks_OD import Attack_OD
        return Attack_OD(cfg, opt_obj)



#--------------------------------------------------------------
#--------------------------------------------------------------
class PatchOPT(object):
    r"""
    class for EOT attacks.
    """
    def __init__(self, 
        cfg, 
        task_interface):

        self.model = task_interface.model
        self.dataset = task_interface.train_loader
        self.device = next(self.model.parameters()).device
        self.cfg = cfg
        self.optloader = task_interface.optloader
        self.task_interface = task_interface
        self.num_patches = cfg["adv_patch"]["num_patches"]
        self.num_iterations = cfg['adv_patch']['patch_opt']['opt_iters']
        self.w_adv = cfg['adv_patch']['patch_opt']['loss']['adv_loss']['factor']
        self.w_real = cfg['adv_patch']['patch_opt']['loss']['smoothness']['factor'], cfg['adv_patch']['patch_opt']['loss']['NPS']['factor'] 
        #self.log_epochs = cfg['adv_patch']['patch_opt']['log_epochs']
        #self.test_epochs = cfg['adv_patch']['patch_opt']['test_epochs']
        #self.save_epochs = cfg['adv_patch']['patch_opt']['save_epochs']


        # Setup optimizer 
        learning_rate = cfg['adv_patch']['patch_opt']["optimizer"]["lr"]
        self.optimizer = torch.optim.Adam(self.model.patches, lr= learning_rate)
        print(self.optimizer)

        # Setup patches params
        self.clipper = patch_utils.PatchesConstraints(set_loader=task_interface.train_loader)
        self.patches_params_array = []
        for _ in range(self.num_patches):
            patch_params = patch_utils.patch_params( 
                noise_magn_percent = 0.02,
                set_loader = self.task_interface.train_loader,
            )
            self.patches_params_array.append(patch_params)
    
        
        self.attack = get_attack(cfg, self)


        #  Setup patch params
        self.p_w, self.real_width, self.offset = cfg['adv_patch']['attr']['width'], cfg['adv_patch']['attr']['world_width'], cfg['adv_patch']['attr']['offset']
        self.block_width, self.rescale = cfg['adv_patch']['attr']['block_width'], cfg['adv_patch']['attr']['rescale']
        self.pixel_width = self.real_width / self.p_w 


        return



    def save_patches(self, i):
        for p in range(self.num_patches):
            patch_png = "patch_%d_" % i + "%d.png" %p 
            patch_pkl = "patch_%d_" % i + "%d.pkl" %p 
            patch_utils.save_patch_numpy(self.model.patches[p], os.path.join(self.task_interface.patches_folder, patch_pkl))
            patch_utils.save_patch_png(self.model.patches[p], os.path.join(self.task_interface.patches_folder, patch_png), set_loader=self.task_interface.train_loader)
        return 



    def sum_gradients_to_patches(self, adv_grad, real_grad):
        for patch_idx, p in enumerate(self.model.patches):
            self.model.patches[patch_idx].grad.data = self.w_adv * adv_grad[patch_idx] + \
                self.w_real[0] * real_grad[patch_idx]['smooth'] + \
                self.w_real[1] * real_grad[patch_idx]['NPS']
        return

    
    def eval(self):
        self.model.eval() 
        self.model.to(self.device)
        self.iter = 0
        self.task_interface.evaluate_patches(self)
        # TODO save results
        return



    def run(self):
        print("-------------Start optimization----------------")

        # train mode is necessary for od models
        if 'od' in self.cfg['task']:
            self.model.train()
        self.model.to(self.device)
        self.iter = None
        i = 0
        cfg = self.cfg

        # save patches
        self.save_patches(i)

        while i <= cfg['adv_patch']['patch_opt']["opt_iters"]:

            self.iter = i
            #-------------------------------------
            # evaluation
            if i % cfg['adv_patch']['patch_opt']["test_log"] == 0:
                self.model.eval() 
                self.model.to(self.device)
                self.task_interface.evaluate_patches(self)
                if 'od' in self.cfg['task']:
                    self.model.train()
                self.model.to(self.device)

            # save patches
            if (i % cfg['adv_patch']['patch_opt']["save_iters"] == 0) or (i == cfg['adv_patch']['patch_opt']["opt_iters"]):
                self.save_patches(i)
            #-------------------------------------
 
            for _, data_items in enumerate(self.optloader):
                images, labels, extrinsic, intrinsic = self.task_interface.item_preprocessing(data_items, self.num_patches)
                images, labels = self.attack.label_prepatching(images, labels)

                # To skip empty labels (object detection)
                if isinstance(labels, list) and len(labels) == 0:
                    continue

                if cfg['adv_patch']['patch_opt']["use_predicted_labels"] is True:
                    labels = self.task_interface.get_predicted_labels(images)

                perturbed_images, patch_masks = self.attack.applied_patches(images, intrinsic, extrinsic)

                for p in range(self.num_patches):
                    self.model.patches[p].requires_grad = True
 
                self.model.apply(self.clipper)
                self.optimizer.zero_grad()
                outputs = self.task_interface.forward(perturbed_images)
                self.curr_outputs, self.curr_labels, self.curr_patch_masks = outputs, labels, patch_masks

                adv_grad_patches = self.attack.compute_adv_grad()
                if adv_grad_patches is None: continue
                physical_grad_patches = self.attack.compute_physical_grad()

                self.sum_gradients_to_patches(adv_grad_patches, physical_grad_patches)
                self.optimizer.step()
                self.model.apply(self.clipper) 

                # cleaning
                torch.cuda.empty_cache()
            
            if i%cfg['adv_patch']['patch_opt']["log_iters"] == 0:
                self.attack.printStat(i)

            

            i += 1
        
        return