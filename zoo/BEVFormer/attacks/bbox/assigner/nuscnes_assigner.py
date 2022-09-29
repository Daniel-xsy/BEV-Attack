from mmdet.core.bbox.builder import BBOX_ASSIGNERS
import numpy as np
import torch


@BBOX_ASSIGNERS.register_module()
class NuScenesAssigner:
    """This class assign each query prediction to a ground truth
    """

    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
            'barrier')

    def __init__(self, dis_thresh=4):
        # TODO: build dis_func, make it more flexible to assign bbox
        self.dis_thresh = dis_thresh


    def get_target(self, bboxes, scores, logits, gt_bboxes):
        """Assign one class pred bbox to conresponding gt
        """

        taken = set()
        match_pair = []

        # no pred or no gt
        if len(gt_bboxes)==0 or len(bboxes)==0:
            return None

        for i, bbox in enumerate(bboxes):

            min_dist = np.inf
            for j, gt_bbox in enumerate(gt_bboxes):

                if j not in taken:
                    dist = np.linalg.norm(np.array(bbox[:2]) - np.array(gt_bbox[:2]))
                    if dist < min_dist:
                        gt_inds = j
                        min_dist = dist
                
            is_match = min_dist < self.dis_thresh

            if is_match:
                taken.add(gt_inds)
                match_pair.append((i, gt_inds))

        match_pair = np.array(match_pair)

        bboxes = bboxes[match_pair[:, 0]]
        scores = scores[match_pair[:, 0]]
        logits = logits[match_pair[:, 0]]
        gt_bbox = gt_bboxes[match_pair[:, 1]]

        return bboxes, scores, logits, gt_bbox


    def assign(self, outputs, gt_bboxes_3d, gt_labels_3d):
        """TODO: add more information
            Args:
                outputs (Tensor): 
                gt_bboxes_3d (Tensor): 
                gt_labels_3d (Tensor): 
            Returns:
                :obj:`Dict`
        """
        
        outputs_bboxes = outputs[0]['pts_bbox']['boxes_3d'].tensor
        outputs_scores = outputs[0]['pts_bbox']['scores_3d']
        outputs_labels = outputs[0]['pts_bbox']['labels_3d']
        outputs_logits = outputs[0]['pts_bbox']['logits_3d']

        gt_bboxes = gt_bboxes_3d[0].data[0][0].tensor
        gt_labels = gt_labels_3d[0].data[0][0]

        targets_pred_bboxes = []
        targets_pred_scores = []
        targets_pred_logits = []
        targets_gt_bboxes = []
        targets_gt_labels = []
        # assign single class
        for i, cls in enumerate(self.CLASSES):

            cls_output_mask = outputs_labels == i
            cls_gt_mask = gt_labels == i

            cls_output_bbox = outputs_bboxes[cls_output_mask]
            cls_outputs_scores = outputs_scores[cls_output_mask]
            cls_outputs_logits = outputs_logits[cls_output_mask]
            cls_gt_bbox = gt_bboxes[cls_gt_mask]

            results = self.get_target(cls_output_bbox, cls_outputs_scores, cls_outputs_logits, cls_gt_bbox)
            if results is not None:
                bbox, scores, logits, gt_bbox = results
                targets_pred_bboxes.append(bbox)
                targets_pred_scores.append(scores)
                targets_pred_logits.append(logits)
                targets_gt_bboxes.append(gt_bbox)
                targets_gt_labels.append((i * torch.ones(len(gt_bbox))).long())

        return {
            'pred_bboxes': torch.cat(targets_pred_bboxes, dim=0),
            'pred_scores': torch.cat(targets_pred_scores, dim=0),
            'pred_logits': torch.cat(targets_pred_logits, dim=0),
            'gt_bbox': torch.cat(targets_gt_bboxes, dim=0),
            'gt_label': torch.cat(targets_gt_labels, dim=0)
        }

