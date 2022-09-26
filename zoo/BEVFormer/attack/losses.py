import torch
from mmdet.core import reduce_mean
from mmcv.utils import TORCH_VERSION, digit_version




def loss_single(self,
                cls_scores,
                bbox_preds,
                gt_bboxes_list,
                gt_labels_list,
                gt_bboxes_ignore_list=None):
    """"Loss function for outputs from a single decoder layer of a single
    feature level.
    Args:
        cls_scores (Tensor): Box score logits from a single decoder layer
            for all images. Shape [bs, num_query, cls_out_channels].
        bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
            for all images, with normalized coordinate (cx, cy, w, h) and
            shape [bs, num_query, 4].
        gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
            with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
        gt_labels_list (list[Tensor]): Ground truth class indices for each
            image with shape (num_gts, ).
        gt_bboxes_ignore_list (list[Tensor], optional): Bounding
            boxes which can be ignored for each image. Default None.
    Returns:
        dict[str, Tensor]: A dictionary of loss components for outputs from
            a single decoder layer.
    """
    num_imgs = cls_scores.size(0)
    cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
    bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
    cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                        gt_bboxes_list, gt_labels_list,
                                        gt_bboxes_ignore_list)
    (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
        num_total_pos, num_total_neg) = cls_reg_targets
    labels = torch.cat(labels_list, 0)
    label_weights = torch.cat(label_weights_list, 0)
    bbox_targets = torch.cat(bbox_targets_list, 0)
    bbox_weights = torch.cat(bbox_weights_list, 0)

    # classification loss
    cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
    # construct weighted avg_factor to match with the official DETR repo
    cls_avg_factor = num_total_pos * 1.0 + \
        num_total_neg * self.bg_cls_weight
    if self.sync_cls_avg_factor:
        cls_avg_factor = reduce_mean(
            cls_scores.new_tensor([cls_avg_factor]))

    cls_avg_factor = max(cls_avg_factor, 1)
    loss_cls = self.loss_cls(
        cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

    # Compute the average number of gt boxes accross all gpus, for
    # normalization purposes
    num_total_pos = loss_cls.new_tensor([num_total_pos])
    num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

    # regression L1 loss
    bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
    normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
    isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
    bbox_weights = bbox_weights * self.code_weights

    loss_bbox = self.loss_bbox(
        bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                            :10], bbox_weights[isnotnan, :10],
        avg_factor=num_total_pos)
    if digit_version(TORCH_VERSION) >= digit_version('1.8'):
        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
    return loss_cls, loss_bbox