Load model checkpoint from ../checkpoints/FCOS3D/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth
## Model Configuration

```
dataset_type = 'CustomNuScenesMonoDataset_Adv'
data_root = '../data/nuscenes/'
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ]),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'attr_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers2d', 'depths'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='LoadImageInfo3D', with_lidar2img=True, with_sensor2lidar=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='RandomFlip3D'),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
                    'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans',
                    'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                    'pts_filename', 'transformation_3d_flow',
                    'sensor2lidar_translation', 'sensor2lidar_rotation'
                ])
        ])
]
eval_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['img'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='CustomNuScenesMonoDataset_Adv',
        data_root='../data/nuscenes/',
        ann_file=
        '../data/nuscenes/nuscenes_infos_temporal_train_mono3d.coco.json',
        img_prefix='../data/nuscenes/',
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        pipeline=[
            dict(type='LoadImageFromFileMono3D'),
            dict(
                type='LoadAnnotations3D',
                with_bbox=True,
                with_label=True,
                with_attr_label=True,
                with_bbox_3d=True,
                with_label_3d=True,
                with_bbox_depth=True),
            dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='Collect3D',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'attr_labels',
                    'gt_bboxes_3d', 'gt_labels_3d', 'centers2d', 'depths'
                ])
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='Camera',
        version='v1.0-mini'),
    val=dict(
        type='CustomNuScenesMonoDataset_Adv',
        data_root='../data/nuscenes/',
        ann_file=
        '../data/nuscenes/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../data/nuscenes/',
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        pipeline=[
            dict(type='LoadImageFromFileMono3D'),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=False),
            dict(
                type='LoadImageInfo3D',
                with_lidar2img=True,
                with_sensor2lidar=True),
            dict(
                type='MultiScaleFlipAug',
                scale_factor=1.0,
                flip=False,
                transforms=[
                    dict(type='RandomFlip3D'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'trailer', 'bus',
                            'construction_vehicle', 'bicycle', 'motorcycle',
                            'pedestrian', 'traffic_cone', 'barrier'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow',
                            'sensor2lidar_translation', 'sensor2lidar_rotation'
                        ])
                ])
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='Camera',
        version='v1.0-mini'),
    test=dict(
        type='CustomNuScenesMonoDataset_Adv',
        data_root='../data/nuscenes/',
        ann_file=
        '../data/nuscenes/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../data/nuscenes/',
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        pipeline=[
            dict(type='LoadImageFromFileMono3D'),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=False),
            dict(
                type='LoadImageInfo3D',
                with_lidar2img=True,
                with_sensor2lidar=True),
            dict(
                type='MultiScaleFlipAug',
                scale_factor=1.0,
                flip=False,
                transforms=[
                    dict(type='RandomFlip3D'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'trailer', 'bus',
                            'construction_vehicle', 'bicycle', 'motorcycle',
                            'pedestrian', 'traffic_cone', 'barrier'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow',
                            'sensor2lidar_translation', 'sensor2lidar_rotation'
                        ])
                ])
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='Camera',
        version='v1.0-mini'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(interval=2)
model = dict(
    type='CustomFCOSMono3D',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1),
        stage_with_dcn=(False, False, True, True)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CustomFCOSMono3DHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        use_direction_classifier=True,
        diff_rad_by_sin=True,
        pred_attrs=True,
        pred_velo=True,
        dir_offset=0.7854,
        strides=[8, 16, 32, 64, 128],
        group_reg_dims=(2, 1, 3, 1, 2),
        cls_branch=(256, ),
        reg_branch=((256, ), (256, ), (256, ), (256, ), ()),
        dir_branch=(256, ),
        attr_branch=(256, ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_attr=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        norm_on_bbox=True,
        centerness_on_reg=True,
        center_sampling=True,
        conv_bias=True,
        dcn_on_last_conv=True,
        train_cfg=None,
        test_cfg=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.8,
            score_thr=0.05,
            min_bbox_size=0,
            max_per_img=200)),
    train_cfg=None,
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.8,
        score_thr=0.05,
        min_bbox_size=0,
        max_per_img=200))
optimizer = dict(
    type='SGD',
    lr=0.002,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
version = 'v1.0-mini'
total_epochs = 12
attack_severity_type = 'initial_const'
attack = dict(
    type='CWAttack',
    max_iterations=50,
    learning_rate=25,
    initial_const=[50],
    img_norm=dict(
        mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
    single_camera=False,
    mono_model=True,
    loss_fn=dict(type='ClassficationObjective', activate=False),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### initial_const 50

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1119    | 0.0730    | 1.0815     | 0.5873     | 1.0157     | 1.0151     | 0.6586     |

Load model checkpoint from ../checkpoints/FCOS3D/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth
## Model Configuration

```
dataset_type = 'CustomNuScenesMonoDataset_Adv'
data_root = '../data/nuscenes/'
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ]),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'attr_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers2d', 'depths'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='LoadImageInfo3D', with_lidar2img=True, with_sensor2lidar=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='RandomFlip3D'),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
                    'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans',
                    'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                    'pts_filename', 'transformation_3d_flow',
                    'sensor2lidar_translation', 'sensor2lidar_rotation'
                ])
        ])
]
eval_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['img'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='CustomNuScenesMonoDataset_Adv',
        data_root='../data/nuscenes/',
        ann_file=
        '../data/nuscenes/nuscenes_infos_temporal_train_mono3d.coco.json',
        img_prefix='../data/nuscenes/',
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        pipeline=[
            dict(type='LoadImageFromFileMono3D'),
            dict(
                type='LoadAnnotations3D',
                with_bbox=True,
                with_label=True,
                with_attr_label=True,
                with_bbox_3d=True,
                with_label_3d=True,
                with_bbox_depth=True),
            dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='Collect3D',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'attr_labels',
                    'gt_bboxes_3d', 'gt_labels_3d', 'centers2d', 'depths'
                ])
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='Camera',
        version='v1.0-mini'),
    val=dict(
        type='CustomNuScenesMonoDataset_Adv',
        data_root='../data/nuscenes/',
        ann_file=
        '../data/nuscenes/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../data/nuscenes/',
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        pipeline=[
            dict(type='LoadImageFromFileMono3D'),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=False),
            dict(
                type='LoadImageInfo3D',
                with_lidar2img=True,
                with_sensor2lidar=True),
            dict(
                type='MultiScaleFlipAug',
                scale_factor=1.0,
                flip=False,
                transforms=[
                    dict(type='RandomFlip3D'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'trailer', 'bus',
                            'construction_vehicle', 'bicycle', 'motorcycle',
                            'pedestrian', 'traffic_cone', 'barrier'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow',
                            'sensor2lidar_translation', 'sensor2lidar_rotation'
                        ])
                ])
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='Camera',
        version='v1.0-mini'),
    test=dict(
        type='CustomNuScenesMonoDataset_Adv',
        data_root='../data/nuscenes/',
        ann_file=
        '../data/nuscenes/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../data/nuscenes/',
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        pipeline=[
            dict(type='LoadImageFromFileMono3D'),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=False),
            dict(
                type='LoadImageInfo3D',
                with_lidar2img=True,
                with_sensor2lidar=True),
            dict(
                type='MultiScaleFlipAug',
                scale_factor=1.0,
                flip=False,
                transforms=[
                    dict(type='RandomFlip3D'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'trailer', 'bus',
                            'construction_vehicle', 'bicycle', 'motorcycle',
                            'pedestrian', 'traffic_cone', 'barrier'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow',
                            'sensor2lidar_translation', 'sensor2lidar_rotation'
                        ])
                ])
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='Camera',
        version='v1.0-mini'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(interval=2)
model = dict(
    type='CustomFCOSMono3D',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1),
        stage_with_dcn=(False, False, True, True)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CustomFCOSMono3DHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        use_direction_classifier=True,
        diff_rad_by_sin=True,
        pred_attrs=True,
        pred_velo=True,
        dir_offset=0.7854,
        strides=[8, 16, 32, 64, 128],
        group_reg_dims=(2, 1, 3, 1, 2),
        cls_branch=(256, ),
        reg_branch=((256, ), (256, ), (256, ), (256, ), ()),
        dir_branch=(256, ),
        attr_branch=(256, ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_attr=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        norm_on_bbox=True,
        centerness_on_reg=True,
        center_sampling=True,
        conv_bias=True,
        dcn_on_last_conv=True,
        train_cfg=None,
        test_cfg=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.8,
            score_thr=0.05,
            min_bbox_size=0,
            max_per_img=200)),
    train_cfg=None,
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.8,
        score_thr=0.05,
        min_bbox_size=0,
        max_per_img=200))
optimizer = dict(
    type='SGD',
    lr=0.002,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
version = 'v1.0-mini'
total_epochs = 12
attack_severity_type = 'initial_const'
attack = dict(
    type='CWAttack',
    max_iterations=50,
    learning_rate=25,
    initial_const=[50],
    img_norm=dict(
        mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
    single_camera=False,
    mono_model=True,
    loss_fn=dict(
        type='LocalizationObjective',
        l2loss=False,
        loc=True,
        vel=True,
        orie=True),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### initial_const 50

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1217    | 0.0632    | 1.0194     | 0.5774     | 0.8748     | 1.4137     | 0.6470     |

