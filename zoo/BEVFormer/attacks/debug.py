import mmcv

file_path = '/home/cihangxie/shaoyuan/BEV-Attack/test.pkl'
data = mmcv.load(file_path)

outputs = data['outputs']
gt_bboxes_3d = data['gt_bboxes_3d']
gt_labels_3d = data['gt_labels_3d']

outputs_bbox = outputs[0]['pts_bbox']['boxes_3d'].tensor
outputs_scores = outputs[0]['pts_bbox']['scores_3d']
outputs_labels = outputs[0]['pts_bbox']['labels_3d']
gt_bbox = gt_bboxes_3d[0].data[0][0].tensor
gt_label = gt_labels_3d[0].data[0][0]

CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
            'barrier')

i = 8
cls = 'traffic_cone'

cls_output_mask = outputs_labels == i
cls_gt_mask = gt_label == i

cls_output_bbox = outputs_bbox[cls_output_mask]
cls_output_conf = outputs_scores[cls_output_mask]
cls_gt_bbox = gt_bbox[cls_gt_mask]

import numpy as np
# for i in range(len(cls_output_conf)):

taken = set()
match_pair = []

for i, bbox in enumerate(cls_output_bbox):

    min_dist = np.inf
    for j, this_bbox in enumerate(cls_gt_bbox):

        if j not in taken:
            dist = np.linalg.norm(np.array(bbox[:2]) - np.array(this_bbox[:2]))
            if dist < min_dist:
                gt_inds = j
                min_dist = dist
        
    is_match = min_dist < 4

    if is_match:
        taken.add(gt_inds)
        match_pair.append((i, gt_inds))

match_pair = np.array(match_pair)

target_bbox = cls_output_bbox[match_pair[:, 0]]
gt_bbox = cls_gt_bbox[match_pair[:, 1]]
a = 1
