'''
Giulio Rossolini
Task interface for Object Detection
'''

import patch_utils
import importlib
import os
from pycocotools.cocoeval import COCOeval
from .task_interfaces import TaskInterface
from torch.utils import data
import numpy as np
import torch


#--------------------------------
# Build Coco Results
def build_coco_results(image_ids, rois, class_ids, scores, resize_factor=(1, 1)):
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    count_imm = 0
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            res_fac = resize_factor[count_imm]
            bbox = np.around(rois[i], 1)

            result = {
                "image_id": int(image_id),
                "category_id": int(class_id), 
                "bbox": [bbox[0] / res_fac[1], bbox[1] / res_fac[0], (bbox[2] - bbox[0]) / res_fac[1], (bbox[3] - bbox[1]) / res_fac[0]],
                "score": score,
            }
            results.append(result)
        count_imm += 1
    return results


#---------------------------------------
# collate fn for object detection
def collate_fn(batch):
        x = [item[0] for item in batch] # put together all images
        y = [item[1] for item in batch] # put together all labels
        z = [item[2] for item in batch] # put together all rescaling_factors
        x = torch.stack(x) # [b,s,n,h,w]
        return x, y, z

#----------------------------------------------------
# Task interface for Semantic Segmentation 
#----------------------------------------------------
class OD_interface(TaskInterface):

    def __init__(self,cfg):
        self.task_utils = importlib.import_module("ptod")
        super().__init__(cfg) 
        return

    def init_loader(self):
         # -------------Setup Dataloader --------------
        cfg = self.cfg
        data_loader = self.task_utils.get_loader(self.cfg["data"]["dataset"])
        data_path = self.cfg["data"]["path"]
        self.opt_obj = None
        self.train_loader = data_loader(
            data_path,
            is_transform=True,
            split=cfg["data"]["train_split"],
            version= cfg["data"]["version"],
            img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
            img_norm = cfg["data"]["img_norm"],
            bgr = cfg["data"]["bgr"],
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
        
        self.n_classes = self.train_loader.n_classes
        
        self.optloader = data.DataLoader(
            self.opt_loader,
            #batch_size= cfg["adv_patch"]['patch_opt']["batch_size_opt"],
            batch_size = 1,
            num_workers= cfg["device"]["n_workers"],
            shuffle=False,
            collate_fn = collate_fn
        )
        self.valloader = data.DataLoader(
            self.validation_loader, 
            #batch_size=cfg["adv_patch"]['patch_opt']["batch_size_val"], 
            batch_size = 1,
            num_workers=cfg["device"]["n_workers"],
            shuffle=False,
            collate_fn = collate_fn
        )

        #---------------- Apply Resize Factor -----------
        if isinstance(self.optloader.dataset, torch.utils.data.Subset):
            train_resize_factors = [None for i in range(0, len(self.optloader.dataset.dataset))]
            self.optloader.dataset.dataset.resize_factors = train_resize_factors
        else:
            train_resize_factors = [None for i in range(0, len(self.optloader.dataset))]
            self.optloader.dataset.resize_factors = train_resize_factors
        if cfg["data"]["img_rows"] is not None:
            if isinstance(self.optloader.dataset, torch.utils.data.Subset):
                coco = self.optloader.dataset.dataset.coco
            else:
                coco = self.optloader.dataset.coco

            if isinstance(self.optloader.dataset, torch.utils.data.Subset):
               opt_dataset = self.optloader.dataset.dataset
            else:
               opt_dataset = self.optloader.dataset

            for i, input_tensor in enumerate(opt_dataset):
                fact = (1,1)
                train_resize_factors[i] = fact
            
            del opt_dataset

        if isinstance(self.optloader.dataset, torch.utils.data.Subset):
            self.optloader.dataset.dataset.resize_factors = train_resize_factors
        else:
            self.optloader.dataset.resize_factors = train_resize_factors


        val_resize_factors = [None for i in range(0, len(self.validation_loader))]
        if isinstance(self.valloader.dataset, torch.utils.data.Subset):
            self.valloader.dataset.dataset.resize_factors = val_resize_factors
        else:
            self.valloader.dataset.resize_factors = val_resize_factors

        if cfg["data"]["img_rows"] is not None:
            if isinstance(self.valloader.dataset, torch.utils.data.Subset):
                coco = self.valloader.dataset.dataset.coco
            else:
                coco = self.valloader.dataset.coco

             # TODO: to work with batchsize > 1 loop through the dataset using batchsize=1 
             # NOTE. assume here a batch_size = 1. Fix this part if you want to extract resize factors with batch > 1
            
            if isinstance(self.valloader.dataset, torch.utils.data.Subset):
               val_dataset = self.valloader.dataset.dataset
            else:
               val_dataset = self.valloader.dataset

            for i, input_tensor in enumerate(val_dataset):
                # _,t,_ = input_tensor
                # t = t[0][0]
                # if len(t) > 0:
                #     im_id = t['image_id']
                #     hh, ww = coco.loadImgs(im_id)[0]['height'], coco.loadImgs(im_id)[0]['width']
                #     fact = (cfg["data"]["img_rows"]/hh, cfg["data"]["img_cols"]/ww)
                #     #print(i, hh, ww)
                # else:
                #     fact = (1, 1)
                fact = (1,1)
                val_resize_factors[i] = fact
            
            del val_dataset
        if isinstance(self.valloader.dataset, torch.utils.data.Subset):
            self.valloader.dataset.dataset.resize_factors = val_resize_factors
        else:
            self.valloader.dataset.resize_factors = val_resize_factors



        # ---------------  Setup Model ------------------
        model_path = cfg["model"]["path"]
        if 'torchvision' not in model_path:
            model_file_name = os.path.split(model_path)[1]
            model_name = model_file_name[: model_file_name.find("_")]
            print(model_name)
        model_file_name = os.path.split(cfg["model"]["path"])[1]
        self.model_name = model_file_name[: model_file_name.find("_")]
        model_dict = {"arch": cfg["model"]["arch"]}
        self.model = self.task_utils.get_model(model_dict, self.n_classes, version=self.cfg["data"]["dataset"])
        if 'torchvision' not in model_path:
            state = torch.load(cfg["model"]["path"], map_location = self.device)
            state = self.task_utils.get_model_state(state, self.model_name)
            self.model.load_state_dict(state)
        self.model = self.model.to(self.device)
        return



    def item_preprocessing(self,data_items, num_patches):
        images, labels, img_resize_factors = data_items
        images = images.to(self.device)
        if isinstance(labels[0], tuple):
            # print(labels)
            extrinsic   = torch.transpose(torch.stack(([torch.stack(x[1]) for x in labels])), 0,1)
            intrinsic   = torch.transpose(torch.stack(([torch.stack(x[2]) for x in labels])), 0,1)
            labels      = [x[0] for x in labels][0]
            # print(extrinsic)
            # print(intrinsic)
            if num_patches is not None:
                extrinsic, intrinsic = [extrinsic[c].to(self.device) for c in range(num_patches)], [intrinsic[c].to(self.device) for c in range(num_patches)]
            else:
                extrinsic, intrinsic = None, None
        else:
            extrinsic, intrinsic = None, None
        self.curr_img_resize_factors = img_resize_factors
        if self.curr_img_resize_factors is None or len(self.curr_img_resize_factors) < 0: 
            self.curr_img_resize_factors = (1,1)
        return images, labels, extrinsic, intrinsic


    def forward(self,x):
        # curr target dict is set in the attack module (for attack opt)
        outputs = self.model(x, self.curr_target_dict)
        return outputs


    def evaluate_patches(self, opt_obj = None, ex_index=None):
        if self.opt_obj is None:
            self.opt_obj = opt_obj

        num_patches = opt_obj.num_patches if opt_obj is not None else None
        summary_results, ex_clear_image, ex_adv_image, ex_clear_out, ex_adv_out, true_label, fact = self.test_loader(
            loader = self.valloader,
            model = self.model, 
            num_patches = num_patches,
            ex_index=ex_index)

        if opt_obj is not None:
            self.test_results.append({
                'epoch': self.opt_obj.iter,
                'results': summary_results
            })
            patch_utils.save_obj(os.path.join(self.cfg["adv_patch"]["path"]["out_dir"], self.cfg["adv_patch"]["path"]["exp_name"], "test_results_%d.pkl" % self.opt_obj.iter), self.test_results)

            patch_utils.save_summary_img_od(
                            ex_clear_image[0], ex_adv_image[0], ex_clear_out, ex_adv_out, true_label,
                            path = os.path.join(self.cfg["adv_patch"]["path"]["out_dir"], self.cfg["adv_patch"]["path"]["exp_name"]),
                            model_name = self.model_name,
                            orig_size =(self.cfg["data"]["img_rows"], self.cfg["data"]["img_cols"]),
                            set_loader = self.validation_loader, 
                            count=self.opt_obj.iter, 
                            img_num=0,
                            factors=fact)

        else:
            self.test_results.append({
                'epoch': 0,
                'results': summary_results
            })
            patch_utils.save_obj(os.path.join(self.cfg["adv_patch"]["path"]["out_dir"], self.cfg["adv_patch"]["path"]["exp_name"], "test_results_eval.pkl"), self.test_results)
            patch_utils.save_images_list([ex_clear_image, true_label, ex_clear_out], self.train_loader, task='2dod', path=os.path.join(self.cfg["adv_patch"]["path"]["out_dir"], self.cfg["adv_patch"]["path"]["exp_name"]))

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
        EX_INDEX = 2 if ex_index is None else ex_index
        saved_ex = False
        self.model.eval()
        self.model.to(self.device)
        results = []
        eval_ids = []
        factors = []
        true_label = None
        fact = (1, 1)
        
    
        if isinstance(loader.dataset, torch.utils.data.Subset):
            data_loader = loader.dataset.dataset
        else:
            data_loader = loader.dataset
        
        for i, data_items in enumerate(loader):
            with torch.no_grad():
                images, labels,extrinsic, intrinsic = self.item_preprocessing(data_items, num_patches)
                if len(labels) > 0:
                    images, labels = self.label_prepatching(images, labels)
                    self.curr_target_dict = [{
                        'boxes': self.labels_boxes,
                        'labels': self.labels_classes,
                        # 'masks': self.labels_segm
                    }]

                    if num_patches is not None:
                        tested_images, patch_masks = self.applied_patches(images, intrinsic, extrinsic)
                        tested_images = tested_images.to(self.device)
                    else:
                        tested_images = images.clone()

                    factors.append(self.curr_img_resize_factors)
                    for resize_fact in self.curr_img_resize_factors:
                        factors.append(resize_fact)

                    # image for visualization
                    if ex_index is not None and i >= ex_index and not saved_ex:
                        ex_clear_image = images[0].clone().unsqueeze(0)
                        ex_clear_out = self.forward(ex_clear_image)
                        true_label = labels.copy()
                        ex_adv_image = tested_images[0].clone().unsqueeze(0)
                        ex_adv_out = self.forward(ex_adv_image)
                        fact = self.curr_img_resize_factors[0]
                        saved_ex = True
                        
                    outputs = self.forward(tested_images)[0]
                    if len(labels) == 0:
                        continue
                    image_id = labels[0]['image_id'] #.cpu().numpy()
                    eval_ids.append(image_id)

                    image_results = build_coco_results([image_id], outputs['boxes'].cpu().numpy(), 
                                                                outputs['labels'].cpu().numpy(), 
                                                                outputs['scores'].cpu().numpy(), 
                                                                resize_factor=self.curr_img_resize_factors)
                    results.extend(image_results)

        coco_results = data_loader.coco.loadRes(results)
        cocoEval = COCOeval(data_loader.coco, coco_results, 'bbox')
        cocoEval.params.imgIds = eval_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        results_list = cocoEval.stats
        summary_results = {
            'mAP @ [IoU=0.50:0.95 | area = all | maxDets = 100]': results_list[0],
            'mAP @ [IoU=0.50 | area = all | maxDets = 100]': results_list[1],
            'mAP @ [IoU=0.75 | area = all | maxDets = 100]': results_list[2],
            'mAP @ [IoU=0.50:0.95 | area = small | maxDets = 100]': results_list[3],
            'mAP @ [IoU=0.50:0.95 | area = medium | maxDets = 100]': results_list[4],
            'mAP @ [IoU=0.50:0.95 | area = large | maxDets = 100]': results_list[5],
            'mAR @ [IoU=0.50:0.95 | area = all | maxDets = 1]': results_list[6],
            'mAR @ [IoU=0.50:0.95 | area = all | maxDets = 10]': results_list[7],
            'mAR @ [IoU=0.50:0.95 | area = all | maxDets = 100]': results_list[8],
            'mAR @ [IoU=0.50:0.95 | area = small | maxDets = 100]': results_list[9],
            'mAR @ [IoU=0.50:0.95 | area = medium | maxDets = 100]': results_list[10],
            'mAR @ [IoU=0.50:0.95 | area = large | maxDets = 100]': results_list[11],
        }
        return summary_results, ex_clear_image, ex_adv_image, ex_clear_out, ex_adv_out, true_label, fact



    #--------------------------
    # Applied patch
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



    #-------------------------
    # Label prepatching
    def label_prepatching(self, images, labels):
        self.labels_classes = torch.tensor([d['category_id'] for d in labels]).to(self.device)
        self.labels_bbox = torch.tensor([d['bbox'] for d in labels]).to(self.device)
        self.labels_boxes = self.labels_bbox.clone()

        self.labels_boxes[:, 2:] += self.labels_boxes[:, :2]
        self.labels_segm = torch.zeros((self.labels_bbox.shape[0], *images.shape[-2:]))
        for bbox_idx in range(self.labels_bbox.shape[0]):
            self.labels_segm[bbox_idx, 
                        int(self.labels_bbox[bbox_idx][1]):int(self.labels_bbox[bbox_idx][1]+self.labels_bbox[bbox_idx][3]),
                        int(self.labels_bbox[bbox_idx][0]):int(self.labels_bbox[bbox_idx][0]+self.labels_bbox[bbox_idx][2])] = 1
        return images, labels
