Load model checkpoint from ../models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth
## Model Configuration

```
dataset_type = 'CustomNuScenesMonoDataset_Adv'
data_root = '../nuscenes_mini/'
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=5,
    is_train=True,
    category_specify=False,
    mono_model=True,
    dataset_cfg=dict(
        dataset=dict(
            type='CustomNuScenesMonoDataset_Adv',
            data_root='../nuscenes/',
            ann_file=
            '../nuscenes/nuscenes_infos_temporal_train_mono3d.coco.json',
            img_prefix='../nuscenes/',
            classes=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
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
                                'construction_vehicle', 'bicycle',
                                'motorcycle', 'pedestrian', 'traffic_cone',
                                'barrier'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                            meta_keys=[
                                'filename', 'ori_shape', 'img_shape',
                                'lidar2img', 'depth2img', 'cam2img',
                                'pad_shape', 'scale_factor', 'flip',
                                'pcd_horizontal_flip', 'pcd_vertical_flip',
                                'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                                'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                                'pcd_rotation', 'pts_filename',
                                'transformation_3d_flow',
                                'sensor2lidar_translation',
                                'sensor2lidar_rotation'
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
            box_type_3d='Camera'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    patch_path=None,
    max_train_samples=18000,
    scale=[0.3],
    patch_size=(200, 200),
    img_norm=dict(
        mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1,
        targets=None),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

Load model checkpoint from ../models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth
## Model Configuration

```
dataset_type = 'CustomNuScenesMonoDataset_Adv'
data_root = '../nuscenes_mini/'
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=5,
    is_train=True,
    category_specify=False,
    mono_model=True,
    dataset_cfg=dict(
        dataset=dict(
            type='CustomNuScenesMonoDataset_Adv',
            data_root='../nuscenes/',
            ann_file=
            '../nuscenes/nuscenes_infos_temporal_train_mono3d.coco.json',
            img_prefix='../nuscenes/',
            classes=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
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
                                'construction_vehicle', 'bicycle',
                                'motorcycle', 'pedestrian', 'traffic_cone',
                                'barrier'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                            meta_keys=[
                                'filename', 'ori_shape', 'img_shape',
                                'lidar2img', 'depth2img', 'cam2img',
                                'pad_shape', 'scale_factor', 'flip',
                                'pcd_horizontal_flip', 'pcd_vertical_flip',
                                'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                                'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                                'pcd_rotation', 'pts_filename',
                                'transformation_3d_flow',
                                'sensor2lidar_translation',
                                'sensor2lidar_rotation'
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
            box_type_3d='Camera'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    patch_path=None,
    max_train_samples=18000,
    scale=[0.3],
    patch_size=(200, 200),
    img_norm=dict(
        mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1,
        targets=None),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1702    | 0.1062    | 0.9421     | 0.4847     | 1.0462     | 1.5371     | 0.4019     |


Load model checkpoint from ../models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth
## Model Configuration

```
dataset_type = 'CustomNuScenesMonoDataset_Adv'
data_root = '../nuscenes_mini/'
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=5,
    is_train=False,
    category_specify=False,
    mono_model=True,
    dataset_cfg=dict(
        dataset=dict(
            type='CustomNuScenesMonoDataset_Adv',
            data_root='../nuscenes/',
            ann_file=
            '../nuscenes/nuscenes_infos_temporal_train_mono3d.coco.json',
            img_prefix='../nuscenes/',
            classes=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
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
                                'construction_vehicle', 'bicycle',
                                'motorcycle', 'pedestrian', 'traffic_cone',
                                'barrier'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                            meta_keys=[
                                'filename', 'ori_shape', 'img_shape',
                                'lidar2img', 'depth2img', 'cam2img',
                                'pad_shape', 'scale_factor', 'flip',
                                'pcd_horizontal_flip', 'pcd_vertical_flip',
                                'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                                'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                                'pcd_rotation', 'pts_filename',
                                'transformation_3d_flow',
                                'sensor2lidar_translation',
                                'sensor2lidar_rotation'
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
            box_type_3d='Camera'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    patch_path=
    '/home/cixie/shaoyuan/BEV-Attack/mmdet_adv/uni_patch/size200_scale0.3_sample18000_PGDMono3D_lr5.pkl',
    max_train_samples=18000,
    scale=[0.3],
    patch_size=(200, 200),
    img_norm=dict(
        mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1,
        targets=None),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1878    | 0.1278    | 0.9035     | 0.5112     | 0.9096     | 1.4711     | 0.4370     |

Load model checkpoint from ../models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth
## Model Configuration

```
dataset_type = 'CustomNuScenesMonoDataset_Adv'
data_root = '../nuscenes_mini/'
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=5,
    is_train=False,
    category_specify=False,
    mono_model=True,
    dataset_cfg=dict(
        dataset=dict(
            type='CustomNuScenesMonoDataset_Adv',
            data_root='../nuscenes/',
            ann_file=
            '../nuscenes/nuscenes_infos_temporal_train_mono3d.coco.json',
            img_prefix='../nuscenes/',
            classes=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
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
                                'construction_vehicle', 'bicycle',
                                'motorcycle', 'pedestrian', 'traffic_cone',
                                'barrier'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                            meta_keys=[
                                'filename', 'ori_shape', 'img_shape',
                                'lidar2img', 'depth2img', 'cam2img',
                                'pad_shape', 'scale_factor', 'flip',
                                'pcd_horizontal_flip', 'pcd_vertical_flip',
                                'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                                'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                                'pcd_rotation', 'pts_filename',
                                'transformation_3d_flow',
                                'sensor2lidar_translation',
                                'sensor2lidar_rotation'
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
            box_type_3d='Camera'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    patch_path=
    '/home/cixie/shaoyuan/BEV-Attack/mmdet_adv/uni_patch/size200_scale0.3_sample3000_lr5_Petr3D_Adv.pkl',
    max_train_samples=18000,
    scale=[0.3],
    patch_size=(200, 200),
    img_norm=dict(
        mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1,
        targets=None),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2100    | 0.1483    | 0.8666     | 0.4962     | 0.8401     | 1.4763     | 0.4383     |

Load model checkpoint from ../models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth
## Model Configuration

```
dataset_type = 'CustomNuScenesMonoDataset_Adv'
data_root = '../nuscenes_mini/'
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=5,
    is_train=False,
    category_specify=False,
    mono_model=True,
    dataset_cfg=dict(
        dataset=dict(
            type='CustomNuScenesMonoDataset_Adv',
            data_root='../nuscenes/',
            ann_file=
            '../nuscenes/nuscenes_infos_temporal_train_mono3d.coco.json',
            img_prefix='../nuscenes/',
            classes=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
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
                                'construction_vehicle', 'bicycle',
                                'motorcycle', 'pedestrian', 'traffic_cone',
                                'barrier'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                            meta_keys=[
                                'filename', 'ori_shape', 'img_shape',
                                'lidar2img', 'depth2img', 'cam2img',
                                'pad_shape', 'scale_factor', 'flip',
                                'pcd_horizontal_flip', 'pcd_vertical_flip',
                                'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                                'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                                'pcd_rotation', 'pts_filename',
                                'transformation_3d_flow',
                                'sensor2lidar_translation',
                                'sensor2lidar_rotation'
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
            box_type_3d='Camera'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    patch_path=
    '/home/cixie/shaoyuan/BEV-Attack/mmdet_adv/uni_patch/size200_scale0.3_sample3000_lr5_Detr3D.pkl',
    max_train_samples=18000,
    scale=[0.3],
    patch_size=(200, 200),
    img_norm=dict(
        mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1,
        targets=None),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2600    | 0.2206    | 0.8230     | 0.4918     | 0.8367     | 1.4974     | 0.3521     |

Load model checkpoint from ../models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth
## Model Configuration

```
dataset_type = 'CustomNuScenesMonoDataset_Adv'
data_root = '../nuscenes_mini/'
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=5,
    is_train=False,
    category_specify=False,
    mono_model=True,
    dataset_cfg=dict(
        dataset=dict(
            type='CustomNuScenesMonoDataset_Adv',
            data_root='../nuscenes/',
            ann_file=
            '../nuscenes/nuscenes_infos_temporal_train_mono3d.coco.json',
            img_prefix='../nuscenes/',
            classes=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
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
                                'construction_vehicle', 'bicycle',
                                'motorcycle', 'pedestrian', 'traffic_cone',
                                'barrier'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                            meta_keys=[
                                'filename', 'ori_shape', 'img_shape',
                                'lidar2img', 'depth2img', 'cam2img',
                                'pad_shape', 'scale_factor', 'flip',
                                'pcd_horizontal_flip', 'pcd_vertical_flip',
                                'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                                'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                                'pcd_rotation', 'pts_filename',
                                'transformation_3d_flow',
                                'sensor2lidar_translation',
                                'sensor2lidar_rotation'
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
            box_type_3d='Camera'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    patch_path=
    '/home/cixie/shaoyuan/BEV-Attack/mmdet_adv/uni_patch/size200_scale0.3_sample3000_BEVFormer_lr5.pkl',
    max_train_samples=18000,
    scale=[0.3],
    patch_size=(200, 200),
    img_norm=dict(
        mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1,
        targets=None),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2469    | 0.2019    | 0.8194     | 0.4906     | 0.8558     | 1.5050     | 0.3750     |

Load model checkpoint from ../models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth
## Model Configuration

```
dataset_type = 'CustomNuScenesMonoDataset_Adv'
data_root = '../nuscenes_mini/'
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=5,
    is_train=False,
    category_specify=False,
    mono_model=True,
    dataset_cfg=dict(
        dataset=dict(
            type='CustomNuScenesMonoDataset_Adv',
            data_root='../nuscenes/',
            ann_file=
            '../nuscenes/nuscenes_infos_temporal_train_mono3d.coco.json',
            img_prefix='../nuscenes/',
            classes=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
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
                                'construction_vehicle', 'bicycle',
                                'motorcycle', 'pedestrian', 'traffic_cone',
                                'barrier'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                            meta_keys=[
                                'filename', 'ori_shape', 'img_shape',
                                'lidar2img', 'depth2img', 'cam2img',
                                'pad_shape', 'scale_factor', 'flip',
                                'pcd_horizontal_flip', 'pcd_vertical_flip',
                                'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                                'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                                'pcd_rotation', 'pts_filename',
                                'transformation_3d_flow',
                                'sensor2lidar_translation',
                                'sensor2lidar_rotation'
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
            box_type_3d='Camera'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    max_train_samples=18000,
    scale=[0.3],
    patch_size=(200, 200),
    img_norm=dict(
        mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1,
        targets=None),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2848    | 0.2428    | 0.7942     | 0.4886     | 0.7515     | 1.4346     | 0.3317     |

Load model checkpoint from ../models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth
## Model Configuration

```
dataset_type = 'CustomNuScenesMonoDataset_Adv'
data_root = '../nuscenes_mini/'
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=10,
    is_train=True,
    category_specify=False,
    mono_model=True,
    dataset_cfg=dict(
        dataset=dict(
            type='CustomNuScenesMonoDataset_Adv',
            data_root='../nuscenes_mini/',
            ann_file=
            '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
            img_prefix='../nuscenes/',
            classes=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
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
                                'construction_vehicle', 'bicycle',
                                'motorcycle', 'pedestrian', 'traffic_cone',
                                'barrier'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                            meta_keys=[
                                'filename', 'ori_shape', 'img_shape',
                                'lidar2img', 'depth2img', 'cam2img',
                                'pad_shape', 'scale_factor', 'flip',
                                'pcd_horizontal_flip', 'pcd_vertical_flip',
                                'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                                'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                                'pcd_rotation', 'pts_filename',
                                'transformation_3d_flow',
                                'sensor2lidar_translation',
                                'sensor2lidar_rotation'
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
            box_type_3d='Camera'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    max_train_samples=1938,
    scale=[0.3],
    patch_size=(100, 100),
    img_norm=dict(
        mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1,
        targets=None),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1473    | 0.0724    | 0.8993     | 0.5706     | 0.8760     | 1.6711     | 0.5427     |

Load model checkpoint from ../models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth
## Model Configuration

```
dataset_type = 'CustomNuScenesMonoDataset_Adv'
data_root = '../nuscenes_mini/'
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=10,
    is_train=False,
    category_specify=False,
    mono_model=True,
    dataset_cfg=dict(
        dataset=dict(
            type='CustomNuScenesMonoDataset_Adv',
            data_root='../nuscenes_mini/',
            ann_file=
            '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
            img_prefix='../nuscenes/',
            classes=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
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
                                'construction_vehicle', 'bicycle',
                                'motorcycle', 'pedestrian', 'traffic_cone',
                                'barrier'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                            meta_keys=[
                                'filename', 'ori_shape', 'img_shape',
                                'lidar2img', 'depth2img', 'cam2img',
                                'pad_shape', 'scale_factor', 'flip',
                                'pcd_horizontal_flip', 'pcd_vertical_flip',
                                'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                                'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                                'pcd_rotation', 'pts_filename',
                                'transformation_3d_flow',
                                'sensor2lidar_translation',
                                'sensor2lidar_rotation'
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
            box_type_3d='Camera'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    max_train_samples=1938,
    scale=[0.3],
    patch_path=[
        '/home/cixie/shaoyuan/BEV-Attack/mmdet_adv/uni_patch/PGDMono3D_coslr_size100_scale0.3_lr10_sample1938.pkl'
    ],
    patch_size=(100, 100),
    img_norm=dict(
        mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1,
        targets=None),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1733    | 0.1069    | 0.9256     | 0.5121     | 0.9068     | 1.5258     | 0.4567     |

Load model checkpoint from ../models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth
## Model Configuration

```
dataset_type = 'CustomNuScenesMonoDataset_Adv'
data_root = '../nuscenes_mini/'
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=10,
    is_train=False,
    category_specify=False,
    mono_model=True,
    dataset_cfg=dict(
        dataset=dict(
            type='CustomNuScenesMonoDataset_Adv',
            data_root='../nuscenes_mini/',
            ann_file=
            '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
            img_prefix='../nuscenes/',
            classes=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
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
                                'construction_vehicle', 'bicycle',
                                'motorcycle', 'pedestrian', 'traffic_cone',
                                'barrier'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                            meta_keys=[
                                'filename', 'ori_shape', 'img_shape',
                                'lidar2img', 'depth2img', 'cam2img',
                                'pad_shape', 'scale_factor', 'flip',
                                'pcd_horizontal_flip', 'pcd_vertical_flip',
                                'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                                'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                                'pcd_rotation', 'pts_filename',
                                'transformation_3d_flow',
                                'sensor2lidar_translation',
                                'sensor2lidar_rotation'
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
            box_type_3d='Camera'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    max_train_samples=1938,
    scale=[0.3],
    patch_path=[
        '/home/cixie/shaoyuan/BEV-Attack/mmdet_adv/uni_patch/Petr3D_Adv_coslr_size100_scale0.3_lr10_sample323.pkl'
    ],
    patch_size=(100, 100),
    img_norm=dict(
        mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1,
        targets=None),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2093    | 0.1510    | 0.8525     | 0.5004     | 0.8534     | 1.5394     | 0.4558     |

Load model checkpoint from ../models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth
## Model Configuration

```
dataset_type = 'CustomNuScenesMonoDataset_Adv'
data_root = '../nuscenes_mini/'
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=10,
    is_train=False,
    category_specify=False,
    mono_model=True,
    dataset_cfg=dict(
        dataset=dict(
            type='CustomNuScenesMonoDataset_Adv',
            data_root='../nuscenes_mini/',
            ann_file=
            '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
            img_prefix='../nuscenes/',
            classes=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
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
                                'construction_vehicle', 'bicycle',
                                'motorcycle', 'pedestrian', 'traffic_cone',
                                'barrier'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                            meta_keys=[
                                'filename', 'ori_shape', 'img_shape',
                                'lidar2img', 'depth2img', 'cam2img',
                                'pad_shape', 'scale_factor', 'flip',
                                'pcd_horizontal_flip', 'pcd_vertical_flip',
                                'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                                'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                                'pcd_rotation', 'pts_filename',
                                'transformation_3d_flow',
                                'sensor2lidar_translation',
                                'sensor2lidar_rotation'
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
            box_type_3d='Camera'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    max_train_samples=1938,
    scale=[0.3],
    patch_path=[
        '/home/cixie/shaoyuan/BEV-Attack/mmdet_adv/uni_patch/Detr3D_coslr_size100_scale0.3_lr10_sample323.pkl'
    ],
    patch_size=(100, 100),
    img_norm=dict(
        mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1,
        targets=None),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2485    | 0.2140    | 0.8372     | 0.4919     | 0.8836     | 1.4809     | 0.3719     |

Load model checkpoint from ../models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth
## Model Configuration

```
dataset_type = 'CustomNuScenesMonoDataset_Adv'
data_root = '../nuscenes_mini/'
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=10,
    is_train=False,
    category_specify=False,
    mono_model=True,
    dataset_cfg=dict(
        dataset=dict(
            type='CustomNuScenesMonoDataset_Adv',
            data_root='../nuscenes_mini/',
            ann_file=
            '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
            img_prefix='../nuscenes/',
            classes=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
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
                                'construction_vehicle', 'bicycle',
                                'motorcycle', 'pedestrian', 'traffic_cone',
                                'barrier'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                            meta_keys=[
                                'filename', 'ori_shape', 'img_shape',
                                'lidar2img', 'depth2img', 'cam2img',
                                'pad_shape', 'scale_factor', 'flip',
                                'pcd_horizontal_flip', 'pcd_vertical_flip',
                                'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                                'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                                'pcd_rotation', 'pts_filename',
                                'transformation_3d_flow',
                                'sensor2lidar_translation',
                                'sensor2lidar_rotation'
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
            box_type_3d='Camera'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    max_train_samples=1938,
    scale=[0.3],
    patch_path=[
        '/home/cixie/shaoyuan/BEV-Attack/mmdet_adv/uni_patch/BEVFormer_coslr_size100_scale0.3_lr10_sample323.pkl'
    ],
    patch_size=(100, 100),
    img_norm=dict(
        mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1,
        targets=None),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2488    | 0.2108    | 0.8385     | 0.4878     | 0.8757     | 1.4904     | 0.3639     |

Load model checkpoint from ../models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth
## Model Configuration

```
dataset_type = 'CustomNuScenesMonoDataset_Adv'
data_root = '../nuscenes_mini/'
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=10,
    is_train=False,
    category_specify=False,
    mono_model=True,
    dataset_cfg=dict(
        dataset=dict(
            type='CustomNuScenesMonoDataset_Adv',
            data_root='../nuscenes_mini/',
            ann_file=
            '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
            img_prefix='../nuscenes/',
            classes=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
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
                                'construction_vehicle', 'bicycle',
                                'motorcycle', 'pedestrian', 'traffic_cone',
                                'barrier'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                            meta_keys=[
                                'filename', 'ori_shape', 'img_shape',
                                'lidar2img', 'depth2img', 'cam2img',
                                'pad_shape', 'scale_factor', 'flip',
                                'pcd_horizontal_flip', 'pcd_vertical_flip',
                                'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                                'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                                'pcd_rotation', 'pts_filename',
                                'transformation_3d_flow',
                                'sensor2lidar_translation',
                                'sensor2lidar_rotation'
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
            box_type_3d='Camera'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    max_train_samples=1938,
    scale=[0.3],
    patch_size=(100, 100),
    img_norm=dict(
        mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1,
        targets=None),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2776    | 0.2379    | 0.8161     | 0.4913     | 0.7689     | 1.4480     | 0.3367     |

Load model checkpoint from ../models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth
## Model Configuration

```
dataset_type = 'CustomNuScenesMonoDataset_Adv'
data_root = '../nuscenes_mini/'
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
        data_root='../nuscenes_mini/',
        ann_file=
        '../nuscenes_mini/nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix='../nuscenes_mini/',
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
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=10,
    is_train=False,
    category_specify=False,
    mono_model=True,
    dataset_cfg=dict(
        dataset=dict(
            type='CustomNuScenesMonoDataset_Adv',
            data_root='../nuscenes_mini/',
            ann_file=
            '../nuscenes_mini/nuscenes_infos_temporal_train_mono3d.coco.json',
            img_prefix='../nuscenes/',
            classes=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
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
                                'construction_vehicle', 'bicycle',
                                'motorcycle', 'pedestrian', 'traffic_cone',
                                'barrier'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                            meta_keys=[
                                'filename', 'ori_shape', 'img_shape',
                                'lidar2img', 'depth2img', 'cam2img',
                                'pad_shape', 'scale_factor', 'flip',
                                'pcd_horizontal_flip', 'pcd_vertical_flip',
                                'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                                'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                                'pcd_rotation', 'pts_filename',
                                'transformation_3d_flow',
                                'sensor2lidar_translation',
                                'sensor2lidar_rotation'
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
            box_type_3d='Camera'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    max_train_samples=1938,
    scale=[0.3],
    patch_size=(100, 100),
    img_norm=dict(
        mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1,
        targets=None),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

