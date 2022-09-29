'''
Federico Nesti
Task interface for 3D stereo object detection
'''

import patch_utils
import importlib
import os
import sys
import numpy as np
import math as m
import subprocess

sys.path.append('pt3dod/models/Stereo-RCNN/lib')
from pt3dod.loader.roibatchLoader import roibatchLoader
from pt3dod.loader.roidb import combined_roidb
from model.utils.config import cfg
from model.rpn.bbox_transform import bbox_transform_inv, kpts_transform_inv, border_transform_inv, clip_boxes
from model.utils.kitti_utils import read_obj_calibration, infer_boundary, write_detection_results
from model.utils.net_utils import vis_detections
from model.utils import box_estimator as box_estimator
from model.utils import vis_3d_utils as vis_utils
from model.roi_layers import nms
from model.dense_align import dense_align


from .task_interfaces import TaskInterface
from torch.utils import data
import torch
from torchvision import transforms as transforms


#----------------------------------------------------
# Task interface for Stereo 3d object detection
#----------------------------------------------------
class stereo3dod_interface(TaskInterface):

    def __init__(self,cfg):
        self.task_utils = importlib.import_module("pt3dod")
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
        
        self.opt_imdb, roidb_opt, ratio_list_opt, ratio_index_opt = combined_roidb(self.opt_loader, False)
        self.opt_loader = roibatchLoader(roidb_opt, ratio_list_opt, ratio_index_opt, 1, \
                          self.opt_imdb, training=True, normalize=False)
        
        self.optloader = data.DataLoader(
            self.opt_loader,
            batch_size= cfg["adv_patch"]['patch_opt']["batch_size_opt"],
            num_workers= cfg["device"]["n_workers"],
            shuffle=False
        )
        
        self.val_imdb, roidb, ratio_list, ratio_index = combined_roidb(self.validation_loader, False)
        self.validation_loader = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                          self.val_imdb, training=False, normalize=False)
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
        self.model = self.task_utils.get_model(model_dict, version=self.cfg["data"]["dataset"])
        state = torch.load(cfg["model"]["path"], map_location = self.device)
        self.uncert = state['uncert']
        state = self.task_utils.get_model_state(state, self.model_name)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device)
        return

    def item_preprocessing(self, data_items, num_patches):
        images_left, images_right, im_info, \
            gt_boxes_left, gt_boxes_right, gt_boxes_merge, gt_dim_orient, gt_kpts, num_boxes, \
                extrinsic_left, intrinsic_left, extrinsic_right, intrinsic_right = data_items
        
        images_left = images_left.to(self.device)
        images_right = images_right.to(self.device)
        im_info = im_info.to(self.device)
        gt_boxes_left = gt_boxes_left.to(self.device)
        gt_boxes_right = gt_boxes_right.to(self.device)
        gt_boxes_merge = gt_boxes_merge.to(self.device)
        gt_dim_orient = gt_dim_orient.to(self.device)
        gt_kpts = gt_kpts.to(self.device)
        num_boxes = num_boxes.to(self.device)

        if num_patches is not None:
            extrinsic_left, intrinsic_left = [extrinsic_left[c].to(self.device) for c in range(num_patches)], [intrinsic_left[c].to(self.device) for c in range(num_patches)]
            extrinsic_right, intrinsic_right = [extrinsic_right[c].to(self.device) for c in range(num_patches)], [intrinsic_right[c].to(self.device) for c in range(num_patches)]
        else:
            extrinsic_left, intrinsic_left, intrinsic_right, extrinsic_right = None, None, None, None

        return (images_left, images_right, im_info), (gt_boxes_left, gt_boxes_right, gt_boxes_merge, gt_dim_orient, gt_kpts, num_boxes), (extrinsic_left, extrinsic_right), (intrinsic_left, intrinsic_right)

    def forward(self, x):
        im_left_data, im_right_data, im_info = x
        gt_boxes_left, gt_boxes_right, gt_boxes_merge, gt_dim_orien, gt_kpts, num_boxes = self.curr_labels
        outputs =\
             self.model(im_left_data, im_right_data, im_info, gt_boxes_left, gt_boxes_right,\
                gt_boxes_merge, gt_dim_orien, gt_kpts, num_boxes)
        return outputs

    def evaluate_patches(self, opt_obj=None, ex_index=None):
        if self.opt_obj is None:
            self.opt_obj = opt_obj

        num_patches = opt_obj.num_patches if opt_obj is not None else None
        summary_results, clear_image, images_clean, images_adv, true_label = self.test_loader(
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
            
            patch_utils.save_summary_img_3dod(
                            images_clean, images_adv,
                            path = os.path.join(self.cfg["adv_patch"]["path"]["out_dir"], self.cfg["adv_patch"]["path"]["exp_name"]),
                            model_name = self.model_name,
                            orig_size =(self.cfg["data"]["img_rows"], self.cfg["data"]["img_cols"]),
                            set_loader = self.validation_loader, 
                            count=self.opt_obj.iter, 
                            img_num=0)

        else:
            self.test_results.append({
                'epoch': 0,
                'results': summary_results
            })
            patch_utils.save_obj(os.path.join(self.cfg["adv_patch"]["path"]["out_dir"], self.cfg["adv_patch"]["path"]["exp_name"], "test_results_eval.pkl"), self.test_results)
            patch_utils.save_images_list([clear_image, true_label, images_adv], self.validation_loader, task='3dod', path=os.path.join(self.cfg["adv_patch"]["path"]["out_dir"], self.cfg["adv_patch"]["path"]["exp_name"]))

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
        self.model.eval()
        self.model.to(self.device)
        results = []
        eval_ids = []
        factors = []

        
        result_dir = os.path.join(self.exp_root, 'results%d' % self.opt_obj.iter) if self.opt_obj is not None else os.path.join(self.exp_root, 'results')
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)    
    
        for i, data_items in enumerate(loader):
            with torch.no_grad():
                images, labels, extrinsic, intrinsic = self.item_preprocessing(data_items, num_patches)
                self.curr_labels = labels
                self.index = i

                if num_patches is not None:
                    tested_images, patch_masks = self.applied_patches(images, intrinsic, extrinsic)
                    tested_images = [im.to(self.device) for im in tested_images]
                else:
                    tested_images = [im.clone() for im in images]

                # image for visualization
                if ex_clear_image is None and i == EX_INDEX:
                    ex_clear_image = images
                    
                    ex_clear_out = self.forward(ex_clear_image)
                    # print(ex_clear_out)
                    im2show_left, im2show_right, _, _, _, _, _, score = self.post_processing(ex_clear_out, ex_clear_image) 
                    true_label = self.show_labels(ex_clear_image, labels)
                    # true_label = labels.copy()
                    ex_adv_image = tested_images
                    ex_adv_out = self.forward(ex_adv_image)
                    # print(ex_adv_out)
                    im2show_left_adv, im2show_right_adv, _, _, _, _, _, score_adv = self.post_processing(ex_adv_out, ex_adv_image)
                    # fact = self.curr_img_resize_factors[0]
                    
                outputs = self.forward(tested_images)
                _, _, calib, box_left, xyz, dim, theta, score = self.post_processing(outputs, tested_images)
                # write result into txt file
                if len(box_left)>0:
                    for box_left_item, xyz_item, dim_item, theta_item, score_item in zip(box_left, xyz, dim, theta, score):
                        write_detection_results(result_dir, '%06d' % i, calib, box_left_item,\
                            xyz_item, dim_item, theta_item, score_item)

        summary_results = {}
        # EVAL DETECTIONS.
        gt_folder = os.path.join(loader.dataset.imdb._kitti_path, loader.dataset.imdb._image_set, 'label_2')
        dt_folder = result_dir
        bashCommand = "./pt3dod/kitti_eval/kitti_eval %s %s" % (gt_folder, dt_folder)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        lines = str(output).split('\\n')
        for line in lines:
            if 'car_detection AP' in line:
                res_str = line.split(':')[1].lstrip().split(' ')
                print(res_str)
                summary_results['detection_results'] = {}
                summary_results['detection_results']['easy'] = float(res_str[0])/100
                summary_results['detection_results']['moderate'] = float(res_str[1])/100
                summary_results['detection_results']['hard'] = float(res_str[2])/100
            elif 'car_orientation AP' in line:
                
                res_str = line.split(':')[1].lstrip().split(' ')
                print(res_str)
                summary_results['orientation_similarity'] = {}
                summary_results['orientation_similarity']['easy'] = float(res_str[0])/100
                summary_results['orientation_similarity']['moderate'] = float(res_str[1])/100
                summary_results['orientation_similarity']['hard'] = float(res_str[2])/100

        print(summary_results)
        return summary_results, ex_clear_image, (im2show_left, im2show_right, score), (im2show_left_adv, im2show_right_adv, score_adv), true_label



    #--------------------------
    # Applied patch
    def applied_patches(self, images, intrinsic, extrinsic):
        images_left, images_right, im_info = images
        perturbed_images_left, patch_masks_left = patch_utils.project_N_patches_batch(
                    images_left.clone(), 
                    self.opt_obj.model.patches, 
                    extrinsic[0], 
                    intrinsic[0],
                    self.opt_obj.patches_params_array,
                    device=self.opt_obj.device,
                    pixel_dim=self.opt_obj.pixel_width, offset=self.opt_obj.offset, rescale=self.opt_obj.rescale
        )
        perturbed_images_right, patch_masks_right = patch_utils.project_N_patches_batch(
                    images_right.clone(), 
                    self.opt_obj.model.patches, 
                    extrinsic[1], 
                    intrinsic[1],
                    self.opt_obj.patches_params_array,
                    device=self.opt_obj.device,
                    pixel_dim=self.opt_obj.pixel_width, offset=self.opt_obj.offset, rescale=self.opt_obj.rescale
        )
        return (perturbed_images_left, perturbed_images_right, im_info), (patch_masks_left, patch_masks_right)

    def labels_prepatching(self, images, labels):
        return images, labels


    def post_processing(self, output, images):
        num_classes = 2
        eval_thresh = 0.05
        vis_thresh = 0.7
        box_left, xyz, theta, dim, score = [], [], [], [], []
        im_info = images[2]

        rois_left, rois_right, cls_prob, bbox_pred, bbox_pred_dim, kpts_prob,\
            left_prob, right_prob, rpn_loss_cls, rpn_loss_box_left_right,\
            RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_dim_orien, RCNN_loss_kpts, rois_label = output
        scores = cls_prob.data
        boxes_left = rois_left.data[:, :, 1:5]
        boxes_right = rois_right.data[:, :, 1:5]

        bbox_pred = bbox_pred.data
        box_delta_left = bbox_pred.new(bbox_pred.size()[1], 4*num_classes).zero_()
        box_delta_right = bbox_pred.new(bbox_pred.size()[1], 4*num_classes).zero_()

        for keep_inx in range(box_delta_left.size()[0]):
            box_delta_left[keep_inx, 0::4] = bbox_pred[0,keep_inx,0::6]
            box_delta_left[keep_inx, 1::4] = bbox_pred[0,keep_inx,1::6]
            box_delta_left[keep_inx, 2::4] = bbox_pred[0,keep_inx,2::6]
            box_delta_left[keep_inx, 3::4] = bbox_pred[0,keep_inx,3::6]

            box_delta_right[keep_inx, 0::4] = bbox_pred[0,keep_inx,4::6]
            box_delta_right[keep_inx, 1::4] = bbox_pred[0,keep_inx,1::6]
            box_delta_right[keep_inx, 2::4] = bbox_pred[0,keep_inx,5::6]
            box_delta_right[keep_inx, 3::4] = bbox_pred[0,keep_inx,3::6]

        box_delta_left = box_delta_left.view(-1,4)
        box_delta_right = box_delta_right.view(-1,4)

        dim_orien = bbox_pred_dim.data
        dim_orien = dim_orien.view(-1,5)

        kpts_prob = kpts_prob.data
        kpts_prob = kpts_prob.view(-1,4*cfg.KPTS_GRID)
        max_prob, kpts_delta = torch.max(kpts_prob,1)

        left_prob = left_prob.data
        left_prob = left_prob.view(-1,cfg.KPTS_GRID)
        _, left_delta = torch.max(left_prob,1)

        right_prob = right_prob.data
        right_prob = right_prob.view(-1,cfg.KPTS_GRID)
        _, right_delta = torch.max(right_prob,1)

        box_delta_left = box_delta_left * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        box_delta_right = box_delta_right * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        dim_orien = dim_orien * torch.FloatTensor(cfg.TRAIN.DIM_NORMALIZE_STDS).cuda() \
                    + torch.FloatTensor(cfg.TRAIN.DIM_NORMALIZE_MEANS).cuda()


        box_delta_left = box_delta_left.view(1,-1,4*num_classes)
        box_delta_right = box_delta_right.view(1, -1,4*num_classes)
        dim_orien = dim_orien.view(1, -1, 5*num_classes)
        kpts_delta = kpts_delta.view(1, -1, 1)
        left_delta = left_delta.view(1, -1, 1)
        right_delta = right_delta.view(1, -1, 1)
        max_prob = max_prob.view(1, -1, 1)

        pred_boxes_left = bbox_transform_inv(boxes_left, box_delta_left, 1)
        pred_boxes_right = bbox_transform_inv(boxes_right, box_delta_right, 1)
        pred_kpts, kpts_type = kpts_transform_inv(boxes_left, kpts_delta,cfg.KPTS_GRID)
        pred_left = border_transform_inv(boxes_left, left_delta,cfg.KPTS_GRID)
        pred_right = border_transform_inv(boxes_left, right_delta,cfg.KPTS_GRID)

        pred_boxes_left = clip_boxes(pred_boxes_left, im_info.data, 1)
        pred_boxes_right = clip_boxes(pred_boxes_right, im_info.data, 1)

        pred_boxes_left /= im_info[0,2].data
        pred_boxes_right /= im_info[0,2].data
        pred_kpts /= im_info[0,2].data
        pred_left /= im_info[0,2].data
        pred_right /= im_info[0,2].data

        scores = scores.squeeze()
        pred_boxes_left = pred_boxes_left.squeeze()
        pred_boxes_right = pred_boxes_right.squeeze()

        pred_kpts = torch.cat((pred_kpts, kpts_type, max_prob, pred_left, pred_right),2)
        pred_kpts = pred_kpts.squeeze()
        dim_orien = dim_orien.squeeze()

        img_path = self.validation_loader.imdb.img_left_path_at(self.index)
        split_path = img_path.split('/')
        image_number = split_path[len(split_path)-1].split('.')[0]
        calib_path = img_path.replace("image_2", "calib")
        calib_path = calib_path.replace("png", "txt")
        calib = read_obj_calibration(calib_path)

        im2show_left = np.uint8(self.validation_loader.imdb.to_image_transform(images[0][0].cpu().numpy(), img_resize=True))
        im2show_right = np.uint8(self.validation_loader.imdb.to_image_transform(images[1][0].cpu().numpy(), img_resize=True))

        for j in range(1, num_classes):
            inds = torch.nonzero(scores[:,j] > eval_thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:,j][inds]
                _, order = torch.sort(cls_scores, 0, True)

                cls_boxes_left = pred_boxes_left[inds][:, j * 4:(j + 1) * 4]
                cls_boxes_right = pred_boxes_right[inds][:, j * 4:(j + 1) * 4]
                cls_dim_orien = dim_orien[inds][:, j * 5:(j + 1) * 5]
                
                cls_kpts = pred_kpts[inds]

                cls_dets_left = torch.cat((cls_boxes_left, cls_scores.unsqueeze(1)), 1)
                cls_dets_right = torch.cat((cls_boxes_right, cls_scores.unsqueeze(1)), 1)

                cls_dets_left = cls_dets_left[order]
                cls_dets_right = cls_dets_right[order]
                cls_dim_orien = cls_dim_orien[order]
                cls_kpts = cls_kpts[order] 

                keep = nms(cls_boxes_left[order, :], cls_scores[order], cfg.TEST.NMS)
                keep = keep.view(-1).long()
                cls_dets_left = cls_dets_left[keep]
                cls_dets_right = cls_dets_right[keep]
                cls_dim_orien = cls_dim_orien[keep]
                cls_kpts = cls_kpts[keep]

                # optional operation, can check the regressed borderline keypoint using 2D box inference
                infered_kpts = infer_boundary(im2show_left.shape, cls_dets_left.cpu().numpy())
                infered_kpts = torch.from_numpy(infered_kpts).type_as(cls_dets_left)
                for detect_idx in range(cls_dets_left.size()[0]):
                    if cls_kpts[detect_idx,4] - cls_kpts[detect_idx,3] < 0.5*(infered_kpts[detect_idx,1]-infered_kpts[detect_idx,0]):
                        cls_kpts[detect_idx,3:5] = infered_kpts[detect_idx]

                im2show_left = vis_detections(im2show_left, 'car', \
                                cls_dets_left.cpu().numpy(), vis_thresh, cls_kpts.cpu().numpy())
                im2show_right = vis_detections(im2show_right, 'car', \
                                cls_dets_right.cpu().numpy(), vis_thresh) 

                # read intrinsic
                f = calib.p2[0,0]
                cx, cy = calib.p2[0,2], calib.p2[1,2]
                bl = (calib.p2[0,3] - calib.p3[0,3])/f

                boxes_all = cls_dets_left.new(0,5)
                kpts_all = cls_dets_left.new(0,5)
                poses_all = cls_dets_left.new(0,8)

                
                for detect_idx in range(cls_dets_left.size()[0]):
                    if cls_dets_left[detect_idx, -1] > eval_thresh:
                        box_left.append(cls_dets_left[detect_idx,0:4].cpu().numpy()) # based on origin image
                        box_right = cls_dets_right[detect_idx,0:4].cpu().numpy() 
                        kpts_u = cls_kpts[detect_idx,0]
                        dim.append(cls_dim_orien[detect_idx,0:3].cpu().numpy())
                        sin_alpha = cls_dim_orien[detect_idx,3]
                        cos_alpha = cls_dim_orien[detect_idx,4]
                        alpha = m.atan2(sin_alpha, cos_alpha)
                        status, state = box_estimator.solve_x_y_z_theta_from_kpt(im2show_left.shape, calib, alpha, \
                                                        dim[-1], box_left[-1], box_right, cls_kpts[detect_idx].cpu().numpy())
                        if status > 0: # not faild
                            poses = images[0].data.new(8).zero_()
                            xyz.append(np.array([state[0], state[1], state[2]]))
                            theta.append(state[3])
                            poses[0], poses[1], poses[2], poses[3], poses[4], poses[5], poses[6], poses[7] = \
                            xyz[-1][0], xyz[-1][1], xyz[-1][2], float(dim[-1][0]), float(dim[-1][1]), float(dim[-1][2]), theta[-1], alpha

                            boxes_all = torch.cat((boxes_all,cls_dets_left[detect_idx,0:5].unsqueeze(0)),0)
                            kpts_all = torch.cat((kpts_all,cls_kpts[detect_idx].unsqueeze(0)),0)
                            poses_all = torch.cat((poses_all,poses.unsqueeze(0)),0)
                
                if boxes_all.dim() > 0:
                    # solve disparity by dense alignment (enlarged image)
                    succ, dis_final = dense_align.align_parallel(calib, im_info.data[0,2], \
                                                        images[0].data, images[1].data, \
                                                        boxes_all[:,0:4], kpts_all, poses_all[:,0:7])
                    
                    # do 3D rectify using the aligned disparity
                    for solved_idx in range(succ.size(0)):
                        if succ[solved_idx] > 0: # succ
                            box_left.append(boxes_all[solved_idx,0:4].cpu().numpy())
                            score.append(boxes_all[solved_idx,4].cpu().numpy())
                            dim.append(poses_all[solved_idx,3:6].cpu().numpy())
                            state_rect, z = box_estimator.solve_x_y_theta_from_kpt(im2show_left.shape, calib, \
                                                            poses_all[solved_idx,7].cpu().numpy(), dim[-1], box_left[-1], \
                                                            dis_final[solved_idx].cpu().numpy(), kpts_all[solved_idx].cpu().numpy())
                            xyz.append(np.array([state_rect[0], state_rect[1], z]))
                            theta.append(state_rect[2])
                            if score[-1] > vis_thresh:
                                # im_box = vis_utils.vis_box_in_bev(im_box, xyz, dim, theta, width=im2show_left.shape[0]*2)
                                im2show_left = vis_utils.vis_single_box_in_img(im2show_left, calib, xyz[-1], dim[-1], theta[-1])

        return im2show_left, im2show_right, calib, box_left, xyz, dim, theta, score


    def show_labels(self, img, labels):
        img_path = self.validation_loader.imdb.img_left_path_at(self.index)
        img = np.uint8(self.validation_loader.imdb.to_image_transform(img[0][0].cpu().numpy(), img_resize=True))
        calib_path = img_path.replace("image_2", "calib")
        calib_path = calib_path.replace("png", "txt")
        calib = read_obj_calibration(calib_path)

        label_path = calib_path.replace('calib', 'label_2')
        with open(label_path, 'r') as f:
            label_lines = f.readlines()

        gt_boxes_left, gt_boxes_right, gt_boxes_merge, gt_dim_orient, gt_kpts, num_boxes = labels
        for i, line in enumerate(label_lines):
            l = line.split()
            lab = l[0]
            if lab.lower() in ('pedestrian', 'car'):
                xyz = [float(x) for x in l[11:14]]
                dim = [float(x) for x in l[8:11]]
                dim[:2] = dim[:2][::-1]
                xyz[1] += dim[1]/2
                theta = float(l[-1])
                print(xyz, dim, theta)
                print( gt_dim_orient[0, i])
                img = vis_utils.vis_single_box_in_img(img, calib, xyz, dim, theta, label=lab.lower())
        return img
