'''
Giulio Rossolini
Object Detection Attacks
'''

import torch
import random
from .attacks import Attack
import patch_utils as patch_utils


def norm(v):
    n = torch.norm(v, p=float('2'))
    return (v/n) if n > 0 else v # to avoid NaN error 

def sign(v):
    v = torch.sign(v)
    return v



#-----------------------------------------------------------
# Object Detection Attack Untargeted
#-----------------------------------------------------------
class Attack_3DOD(Attack):
    r"""
    class for untargeted 2D Object Detection Attack.
    """

    def __init__(self, cfg, opt_obj):
        self.initStat()
        # OPTIMIZATION OBJECTIVE: PUT ALL 4 AS -1 FOR UNTARGETED ALSO ON THE BBOX
        self.opt_sign = [-1, -1, -1, -1, -1, -1]
        self.attacked_class, self.target_class = None, None 
        self.adv_loss_weights = [1, 1, 1.0, 1.0] 
        
        super().__init__(cfg, opt_obj) 

    #------------------------------
    #Init attack stats 
    def initStat(self):
        self.epoch_loss_tot = 0
        self.epoch_loss_info_rpn_cls        = 0
        self.epoch_loss_info_rpn_bbox_lr    = 0
        self.epoch_loss_info_cls            = 0
        self.epoch_loss_info_bbox           = 0
        self.epoch_loss_info_dim_orien      = 0
        self.epoch_loss_info_kpts           = 0
        self.epoch_loss_info_samples        = 0
        self.epoch_loss_info_smoothness     = 0
        self.epoch_loss_info_NPS            = 0
        return


    #------------------------------
    # Print attack stats 
    def printStat(self, i, reset_stat = True):
        fmt_str = "Epochs [{:d}/{:d}]  Mean Losses: total loss {:.4f}, cls loss {:.4f}, bbox loss {:.4f}, obj loss: {:.4f}, rpn_bbox loss {:.4f}, dim_orien loss: {:.4f}, kpts loss: {:.4f}, smooth loss {:.4f}, NPS loss {:.4f},(on {:d} training samples) "
        print_str = fmt_str.format(
                i + 1,
                self.cfg["adv_patch"]['patch_opt']["opt_iters"],
                self.epoch_loss_tot/self.epoch_loss_info_samples,
                self.epoch_loss_info_cls/self.epoch_loss_info_samples,
                self.epoch_loss_info_bbox/self.epoch_loss_info_samples,
                self.epoch_loss_info_rpn_cls/self.epoch_loss_info_samples,
                self.epoch_loss_info_rpn_bbox_lr/self.epoch_loss_info_samples,
                self.epoch_loss_info_dim_orien/self.epoch_loss_info_samples,
                self.epoch_loss_info_kpts/self.epoch_loss_info_samples,
                self.epoch_loss_info_smoothness /self.epoch_loss_info_samples,
                self.epoch_loss_info_NPS /self.epoch_loss_info_samples,
                self.epoch_loss_info_samples
            )
        print(print_str)
        i += 1

        if reset_stat:
            self.initStat()

        return

    #--------------------------
    # Label prepatching
    def label_prepatching(self, images, labels):
        self.opt_obj.task_interface.curr_labels = labels 
        return images, labels


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



    #--------------------------
    # Compute adv grad
    def compute_adv_grad(self):
        outputs = self.opt_obj.curr_outputs
        labels = self.opt_obj.curr_labels
        patch_masks = self.opt_obj.curr_patch_masks

        rois_left, rois_right, cls_prob, bbox_pred, bbox_pred_dim, kpts_prob,\
            left_prob, right_prob, rpn_loss_cls, rpn_loss_box_left_right,\
            RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_dim_orien, RCNN_loss_kpts, rois_label = outputs

        
        uncert = self.opt_obj.task_interface.uncert
       
        tot_loss = rpn_loss_cls.mean() * torch.exp(-uncert[0]) + uncert[0] +\
              rpn_loss_box_left_right.mean() * torch.exp(-uncert[1]) + uncert[1] +\
              RCNN_loss_cls.mean() * torch.exp(-uncert[2]) + uncert[2]+\
              RCNN_loss_bbox.mean() * torch.exp(-uncert[3]) + uncert[3] +\
              RCNN_loss_dim_orien.mean() * torch.exp(-uncert[4]) + uncert[4] +\
              RCNN_loss_kpts.mean() * torch.exp(-uncert[5]) + uncert[5]
        
        self.epoch_loss_tot             += tot_loss.item()
        self.epoch_loss_info_cls        += RCNN_loss_cls.item()
        self.epoch_loss_info_bbox       += RCNN_loss_bbox.item()
        self.epoch_loss_info_rpn_cls        += rpn_loss_cls.item()
        self.epoch_loss_info_rpn_bbox_lr   += rpn_loss_box_left_right.item()
        self.epoch_loss_info_dim_orien  += RCNN_loss_dim_orien.item()
        self.epoch_loss_info_kpts       += RCNN_loss_kpts.item()
        self.epoch_loss_info_samples    += patch_masks[0].shape[0]
 
        loss_array =  [rpn_loss_box_left_right, RCNN_loss_bbox, RCNN_loss_dim_orien, RCNN_loss_kpts] 

        retain_graph_bool = True
        norm_grad_losses = []
        for _ in range(self.opt_obj.num_patches):
            norm_grad_losses.append([None] * len(loss_array))

        
        # compute and normalize each loss
        for count, l in enumerate(loss_array):
            self.opt_obj.optimizer.zero_grad()
            l.backward(retain_graph=retain_graph_bool)
            for patch_idx, p in enumerate(self.opt_obj.model.patches):
                grad_loss = self.opt_obj.model.patches[patch_idx].grad.data.clone().to(self.opt_obj.device)
                norm_grad_losses[patch_idx][count] = norm(grad_loss)

        adv_grad_patches = []
        for patch_idx, p in enumerate(self.opt_obj.model.patches):
            final_grad_adv = 0
            for w_idx, w in enumerate(self.adv_loss_weights):
                final_grad_adv += w * (self.opt_sign[w_idx] * norm_grad_losses[patch_idx][w_idx]) 
            adv_grad_patches.append(norm(final_grad_adv))
        return adv_grad_patches




    def post_processing(self, output, images):
        num_classes = 2
        eval_thresh = 0.05
        vis_thresh = 0.
        box_left, xyz, theta, dim, score = None, None, None, None, None

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

        # det_toc = time.time()
        # detect_time = det_toc - det_tic
        # print(detect_time)

        img_path = self.validation_loader.imdb.img_left_path_at(self.index)
        split_path = img_path.split('/')
        image_number = split_path[len(split_path)-1].split('.')[0]
        calib_path = img_path.replace("image_2", "calib")
        calib_path = calib_path.replace("png", "txt")
        calib = read_obj_calibration(calib_path)

        im2show_left = np.uint8(self.validation_loader.imdb.to_image_transform(images[0][0].cpu().numpy()))
        im2show_right = np.uint8(self.validation_loader.imdb.to_image_transform(images[1][0].cpu().numpy()))

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
                        box_left = cls_dets_left[detect_idx,0:4].cpu().numpy()  # based on origin image
                        box_right = cls_dets_right[detect_idx,0:4].cpu().numpy() 
                        kpts_u = cls_kpts[detect_idx,0]
                        dim = cls_dim_orien[detect_idx,0:3].cpu().numpy()
                        sin_alpha = cls_dim_orien[detect_idx,3]
                        cos_alpha = cls_dim_orien[detect_idx,4]
                        alpha = m.atan2(sin_alpha, cos_alpha)
                        status, state = box_estimator.solve_x_y_z_theta_from_kpt(im2show_left.shape, calib, alpha, \
                                                        dim, box_left, box_right, cls_kpts[detect_idx].cpu().numpy())
                        if status > 0: # not faild
                            poses = images[0].data.new(8).zero_()
                            xyz = np.array([state[0], state[1], state[2]])
                            theta = state[3]
                            poses[0], poses[1], poses[2], poses[3], poses[4], poses[5], poses[6], poses[7] = \
                            xyz[0], xyz[1], xyz[2], float(dim[0]), float(dim[1]), float(dim[2]), theta, alpha

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
                            box_left = boxes_all[solved_idx,0:4].cpu().numpy()
                            score = boxes_all[solved_idx,4].cpu().numpy()
                            dim = poses_all[solved_idx,3:6].cpu().numpy()
                            state_rect, z = box_estimator.solve_x_y_theta_from_kpt(im2show_left.shape, calib, \
                                                            poses_all[solved_idx,7].cpu().numpy(), dim, box_left, \
                                                            dis_final[solved_idx].cpu().numpy(), kpts_all[solved_idx].cpu().numpy())
                            xyz = np.array([state_rect[0], state_rect[1], z])
                            theta = state_rect[2]

                            if score > vis_thresh:
                                # im_box = vis_utils.vis_box_in_bev(im_box, xyz, dim, theta, width=im2show_left.shape[0]*2)
                                im2show_left = vis_utils.vis_single_box_in_img(im2show_left, calib, xyz, dim, theta)

        return im2show_left, im2show_right, calib, box_left, xyz, dim, theta, score