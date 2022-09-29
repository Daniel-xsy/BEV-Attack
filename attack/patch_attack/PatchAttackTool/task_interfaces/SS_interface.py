'''
Giulio Rossolini
Task interface for Semantic Segmentation 
'''

import patch_utils
import importlib
import os

from .task_interfaces import TaskInterface
from torch.utils import data
import torch
import matplotlib.pyplot as plt


import patch_utils as patch_utils
from ptsemseg.metrics import runningScore

#----------------------------------------------------
# Task interface for Semantic Segmentation 
#----------------------------------------------------
class SS_interface(TaskInterface):

    def __init__(self,cfg):
        self.task_utils = importlib.import_module("ptsemseg")
        super().__init__(cfg) 
        return


    def init_loader(self):
         # -------------Setup Dataloader --------------
        cfg = self.cfg
        self.opt_obj = None
        data_loader = self.task_utils.get_loader(self.cfg["data"]["dataset"])
        data_path = self.cfg["data"]["path"]
        self.train_loader = data_loader(
            data_path,
            is_transform=True,
            split=cfg["data"]["train_split"],
            version= cfg["data"]["version"],
            img_size=(cfg["data"]["img_rows"], self.cfg["data"]["img_cols"]),
            img_norm = cfg["data"]["img_norm"],
            bgr = self.cfg["data"]["bgr"],
            std_version = cfg["data"]["std_version"], 
            bottom_crop = 0,
            num_patches = cfg["adv_patch"]["num_patches"]
        )
        
        num_train_samples = cfg['adv_patch']['patch_opt']['num_opt_samples']
        if num_train_samples is not None:
            self.opt_loader, _ = torch.utils.data.random_split(
                self.train_loader, 
                [num_train_samples, len(self.train_loader)-num_train_samples])
        else:
            self.opt_loader = self.train_loader
        print("num optimization images (from training set): " + str(len(self.opt_loader)))
        
        self.validation_loader = data_loader(
            data_path,
            is_transform=True,
            split=cfg["data"]["val_split"],
            version= cfg["data"]["version"],
            img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
            img_norm = cfg["data"]["img_norm"],
            bgr = cfg["data"]["bgr"], 
            std_version = cfg["data"]["std_version"],
            bottom_crop = 0,
            num_patches = cfg["adv_patch"]["num_patches"]
        )

        print(len(self.validation_loader))
        
        self.n_classes = self.train_loader.n_classes
        
        self.optloader = data.DataLoader(
            self.opt_loader,
            batch_size= cfg["adv_patch"]['patch_opt']["batch_size_opt"],
            num_workers= cfg["device"]["n_workers"],
            shuffle=True
        )
        self.valloader = data.DataLoader(
            self.validation_loader, 
            batch_size=cfg["adv_patch"]['patch_opt']["batch_size_val"], 
            num_workers=cfg["device"]["n_workers"],
            shuffle=False
        )

        # ---------------  Setup Model ------------------
        model_file_name = os.path.split(cfg["model"]["path"])[1]
        self.model_name = model_file_name[: model_file_name.find("_")]
        model_dict = {"arch": cfg["model"]["arch"]}
        self.model = self.task_utils.get_model(model_dict, self.n_classes, version=self.cfg["data"]["dataset"])
        state = torch.load(cfg["model"]["path"], map_location = self.device)
        state = self.task_utils.get_model_state(state, self.model_name)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device)
        return

    def item_preprocessing(self,data_items, num_patches):
        images, labels = data_items
        images = images.to(self.device)

        if isinstance(labels, tuple) or type(labels) == list:
            (labels, extrinsic, intrinsic) = labels
            if num_patches is not None:
                extrinsic, intrinsic = [extrinsic[c].to(self.device) for c in range(num_patches)], [intrinsic[c].to(self.device) for c in range(num_patches)]
            else:
                extrinsic, intrinsic = None, None
        else:
            extrinsic, intrinsic = None, None
        labels = labels.to(self.device)
        return images, labels, extrinsic, intrinsic

    def forward(self,x):
        outputs = self.model(x)
        return outputs

    def evaluate_patches(self, opt_obj=None, ex_index=None):
        if self.opt_obj is None:
            self.opt_obj = opt_obj

        num_patches = opt_obj.num_patches if opt_obj is not None else None
        summary_results, ex_clear_image, ex_adv_image, ex_clear_out, ex_adv_out, true_label = self.test_loader(
            loader = self.valloader,
            model = self.model, 
            num_patches=num_patches,
            ex_index=ex_index)
        if opt_obj is not None:
            self.test_results.append({
                'epoch': self.opt_obj.iter,
                'results': summary_results
            })
            
            patch_utils.save_obj(os.path.join(self.cfg["adv_patch"]["path"]["out_dir"], self.cfg["adv_patch"]["path"]["exp_name"], "test_results_%d.pkl" % self.opt_obj.iter), self.test_results)

            patch_utils.save_summary_img(
                tensor_list = [ex_adv_image[0], ex_clear_image[0], ex_adv_out, ex_clear_out], 
                path = os.path.join(self.cfg["adv_patch"]["path"]["out_dir"], self.cfg["adv_patch"]["path"]["exp_name"]),
                model_name = self.model_name,
                orig_size =(self.cfg["data"]["img_rows"], self.cfg["data"]["img_cols"]),
                set_loader = self.train_loader, 
                count=opt_obj.iter, 
                img_num=0)
        else:
            self.test_results.append({
                'epoch': 0,
                'results': summary_results
            })
            patch_utils.save_obj(os.path.join(self.cfg["adv_patch"]["path"]["out_dir"], self.cfg["adv_patch"]["path"]["exp_name"], "test_results_eval.pkl"), self.test_results)
            patch_utils.save_images_list([ex_clear_image, true_label, ex_clear_out], self.train_loader, task='ss', path=os.path.join(self.cfg["adv_patch"]["path"]["out_dir"], self.cfg["adv_patch"]["path"]["exp_name"]))
        
        print(summary_results)
        return

    def evaluate(self):
        raise NotImplementedError 

    def test_loader( 
                self, 
                loader, 
                model,
                num_patches=None,
                ex_index=None):
        ex_clear_image, ex_adv_image      =  None, None
        ex_clear_out, ex_adv_out          =  None, None
        EX_INDEX = 1 if ex_index is None else ex_index
        self.model.eval()
        self.model.to(self.device)
    
        
        running_metrics = runningScore(self.n_classes)

        for i, data_items in enumerate(loader):
            with torch.no_grad():
                images, labels,extrinsic, intrinsic = self.item_preprocessing(data_items, num_patches)

                if num_patches is not None:
                    tested_images, patch_masks = self.applied_patches(images, intrinsic, extrinsic)
                    tested_images = tested_images.to(self.device)
                else:
                    tested_images = images.clone()

                # image for visualization
                if ex_clear_image is None and i == EX_INDEX:
                    ex_clear_image = images[0].clone().unsqueeze(0)
                    ex_clear_out = self.forward(ex_clear_image)
                    ex_adv_image = tested_images[0].clone().unsqueeze(0)
                    ex_adv_out = self.forward(ex_adv_image)
                    true_label = labels[0].clone()

                outputs = self.forward(tested_images)
                pred = outputs.data.max(1)[1].cpu().numpy()
                gt = labels

                if num_patches is not None:
                    gt = patch_utils.remove_mask(labels, patch_masks.detach())
                    pred = patch_utils.remove_mask(pred, patch_masks.detach())
                running_metrics.update(gt.cpu().numpy(), pred)


        summary_results = running_metrics.get_scores()
        return summary_results, ex_clear_image, ex_adv_image, ex_clear_out, ex_adv_out, true_label

    
    def applied_patches(self, images, intrinsic, extrinsic):
        perturbed_images, patch_masks = patch_utils.project_N_patches_batch(
                    images.clone(), 
                    self.opt_obj.model.patches, 
                    extrinsic, 
                    intrinsic,
                    self.opt_obj.patches_params_array,
                    device=self.opt_obj.device,
                    pixel_dim=self.opt_obj.pixel_width, offset=self.opt_obj.offset, rescale=self.opt_obj.rescale
        )
        return perturbed_images, patch_masks


    def get_predicted_labels(self, images):
        # extract the predicted labels that are used in the optimization loss function
        prediction_labels = None
        with torch.no_grad():
            clear_prediction = self.model(images)
            
            if not isinstance(clear_prediction, tuple):
                aus = clear_prediction.transpose(1, 2).transpose(2, 3).contiguous().view(-1, self.n_classes)
                prediction_labels = torch.argmax(aus, dim=1).to(self.device)
            else:
                prediction_labels = []
                for _, pred in enumerate(clear_prediction) :
                    aus = pred.transpose(1, 2).transpose(2, 3).contiguous().view(-1, self.n_classes)
                    prediction_labels.append(torch.argmax(aus, dim=1).to(self.device))
                prediction_labels = tuple(prediction_labels)
        return prediction_labels