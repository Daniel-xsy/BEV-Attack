Load model checkpoint from /home/cixie/shaoyuan/BEV-Attack/models/bevdet/bevdet-r101.pth
## Model Configuration

```
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDataset'
data_root = '../../nuscenes_mini/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        is_train=True,
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                   'transformation_3d_flow', 'img_info'))
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['img_inputs'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        dataset=dict(
            type='NuScenesDataset',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    is_train=True,
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    dummy=True,
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.3925, 0.3925],
                    scale_ratio_range=[0.95, 1.05],
                    translation_std=[0, 0, 0],
                    update_img2lidar=True),
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.5,
                    update_img2lidar=True),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                dict(
                    type='ObjectNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='Collect3D',
                    keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'lidar2img', 'depth2img', 'cam2img',
                               'pad_shape', 'scale_factor', 'flip',
                               'pcd_horizontal_flip', 'pcd_vertical_flip',
                               'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                               'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                               'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'img_info'))
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            test_mode=False,
            use_valid_flag=True,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            box_type_3d='LiDAR',
            img_info_prototype='bevdet')),
    val=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'),
    test=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
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
data_config = dict(
    cams=[
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.04)
grid_config = dict(
    xbound=[-51.2, 51.2, 0.8],
    ybound=[-51.2, 51.2, 0.8],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[1.0, 60.0, 1.0])
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 64
model = dict(
    type='BEVDet_Adv',
    img_backbone=dict(
        pretrained='torchvision://resnet101',
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPNForBEVDet',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShoot',
        grid_config=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04),
        use_bev_pool=False,
        numC_Trans=64),
    img_bev_encoder_backbone=dict(type='ResNetForBEVDet', numC_input=64),
    img_bev_encoder_neck=dict(
        type='FPN_LSS', in_channels=640, out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead_Adv',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder_Adv',
            pc_range=[-51.2, -51.2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=9),
        separate_head=dict(
            type='SeparateHead',
            init_bias=-2.19,
            final_kernel=3,
            in_channels=64,
            heads=dict(
                reg=(2, 2),
                height=(1, 2),
                dim=(3, 2),
                rot=(2, 2),
                vel=(2, 2),
                heatmap=(2, 2)),
            num_cls=2),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
        train_cfg=None,
        test_cfg=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    train_cfg=None,
    test_cfg=dict(
        pts=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    pretrained=None)
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
img_norm_cfg = dict(
    mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False)
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=10,
    is_train=False,
    category_specify=False,
    totensor=True,
    dataset_cfg=dict(
        dataset=dict(
            type='NuScenesDataset',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='MultiScaleFlipAug3D',
                    img_scale=(1333, 800),
                    pts_scale_ratio=1,
                    flip=False,
                    transforms=[
                        dict(
                            type='DefaultFormatBundle3D',
                            class_names=[
                                'car', 'truck', 'construction_vehicle', 'bus',
                                'trailer', 'barrier', 'motorcycle', 'bicycle',
                                'pedestrian', 'traffic_cone'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=[
                                'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'
                            ])
                    ])
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            filter_empty_gt=False,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            img_info_prototype='bevdet'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    scale=[0.3],
    max_train_samples=323,
    patch_size=(100, 100),
    img_norm=dict(
        mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2670    | 0.1924    | 0.8274     | 0.4888     | 0.7739     | 0.8347     | 0.3671     |

Load model checkpoint from /home/cixie/shaoyuan/BEV-Attack/models/bevdet/bevdet-r101.pth
## Model Configuration

```
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDataset'
data_root = '../../nuscenes_mini/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        is_train=True,
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                   'transformation_3d_flow', 'img_info'))
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['img_inputs'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        dataset=dict(
            type='NuScenesDataset',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    is_train=True,
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    dummy=True,
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.3925, 0.3925],
                    scale_ratio_range=[0.95, 1.05],
                    translation_std=[0, 0, 0],
                    update_img2lidar=True),
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.5,
                    update_img2lidar=True),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                dict(
                    type='ObjectNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='Collect3D',
                    keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'lidar2img', 'depth2img', 'cam2img',
                               'pad_shape', 'scale_factor', 'flip',
                               'pcd_horizontal_flip', 'pcd_vertical_flip',
                               'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                               'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                               'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'img_info'))
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            test_mode=False,
            use_valid_flag=True,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            box_type_3d='LiDAR',
            img_info_prototype='bevdet')),
    val=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'),
    test=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
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
data_config = dict(
    cams=[
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.04)
grid_config = dict(
    xbound=[-51.2, 51.2, 0.8],
    ybound=[-51.2, 51.2, 0.8],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[1.0, 60.0, 1.0])
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 64
model = dict(
    type='BEVDet_Adv',
    img_backbone=dict(
        pretrained='torchvision://resnet101',
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPNForBEVDet',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShoot',
        grid_config=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04),
        use_bev_pool=False,
        numC_Trans=64),
    img_bev_encoder_backbone=dict(type='ResNetForBEVDet', numC_input=64),
    img_bev_encoder_neck=dict(
        type='FPN_LSS', in_channels=640, out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead_Adv',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder_Adv',
            pc_range=[-51.2, -51.2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=9),
        separate_head=dict(
            type='SeparateHead',
            init_bias=-2.19,
            final_kernel=3,
            in_channels=64,
            heads=dict(
                reg=(2, 2),
                height=(1, 2),
                dim=(3, 2),
                rot=(2, 2),
                vel=(2, 2),
                heatmap=(2, 2)),
            num_cls=2),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
        train_cfg=None,
        test_cfg=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    train_cfg=None,
    test_cfg=dict(
        pts=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    pretrained=None)
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
img_norm_cfg = dict(
    mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False)
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=10,
    is_train=True,
    category_specify=False,
    totensor=True,
    dataset_cfg=dict(
        dataset=dict(
            type='NuScenesDataset',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='MultiScaleFlipAug3D',
                    img_scale=(1333, 800),
                    pts_scale_ratio=1,
                    flip=False,
                    transforms=[
                        dict(
                            type='DefaultFormatBundle3D',
                            class_names=[
                                'car', 'truck', 'construction_vehicle', 'bus',
                                'trailer', 'barrier', 'motorcycle', 'bicycle',
                                'pedestrian', 'traffic_cone'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=[
                                'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'
                            ])
                    ])
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            filter_empty_gt=False,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            img_info_prototype='bevdet'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    scale=[0.3],
    max_train_samples=323,
    patch_size=(100, 100),
    img_norm=dict(
        mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

Load model checkpoint from /home/cixie/shaoyuan/BEV-Attack/models/bevdet/bevdet-r101.pth
## Model Configuration

```
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDataset_Adv'
data_root = '../../nuscenes_mini/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        is_train=True,
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                   'transformation_3d_flow', 'img_info'))
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['img_inputs'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    is_train=True,
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    dummy=True,
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.3925, 0.3925],
                    scale_ratio_range=[0.95, 1.05],
                    translation_std=[0, 0, 0],
                    update_img2lidar=True),
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.5,
                    update_img2lidar=True),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                dict(
                    type='ObjectNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='Collect3D',
                    keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'lidar2img', 'depth2img', 'cam2img',
                               'pad_shape', 'scale_factor', 'flip',
                               'pcd_horizontal_flip', 'pcd_vertical_flip',
                               'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                               'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                               'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'img_info'))
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            test_mode=False,
            use_valid_flag=True,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            box_type_3d='LiDAR',
            img_info_prototype='bevdet')),
    val=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'),
    test=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
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
data_config = dict(
    cams=[
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.04)
grid_config = dict(
    xbound=[-51.2, 51.2, 0.8],
    ybound=[-51.2, 51.2, 0.8],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[1.0, 60.0, 1.0])
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 64
model = dict(
    type='BEVDet_Adv',
    img_backbone=dict(
        pretrained='torchvision://resnet101',
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPNForBEVDet',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShoot',
        grid_config=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04),
        use_bev_pool=False,
        numC_Trans=64),
    img_bev_encoder_backbone=dict(type='ResNetForBEVDet', numC_input=64),
    img_bev_encoder_neck=dict(
        type='FPN_LSS', in_channels=640, out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead_Adv',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder_Adv',
            pc_range=[-51.2, -51.2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=9),
        separate_head=dict(
            type='SeparateHead',
            init_bias=-2.19,
            final_kernel=3,
            in_channels=64,
            heads=dict(
                reg=(2, 2),
                height=(1, 2),
                dim=(3, 2),
                rot=(2, 2),
                vel=(2, 2),
                heatmap=(2, 2)),
            num_cls=2),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
        train_cfg=None,
        test_cfg=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    train_cfg=None,
    test_cfg=dict(
        pts=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    pretrained=None)
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
img_norm_cfg = dict(
    mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False)
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=10,
    is_train=True,
    category_specify=False,
    totensor=True,
    dataset_cfg=dict(
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='MultiScaleFlipAug3D',
                    img_scale=(1333, 800),
                    pts_scale_ratio=1,
                    flip=False,
                    transforms=[
                        dict(
                            type='DefaultFormatBundle3D',
                            class_names=[
                                'car', 'truck', 'construction_vehicle', 'bus',
                                'trailer', 'barrier', 'motorcycle', 'bicycle',
                                'pedestrian', 'traffic_cone'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=[
                                'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'
                            ])
                    ])
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            filter_empty_gt=False,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            img_info_prototype='bevdet'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    scale=[0.3],
    max_train_samples=323,
    patch_size=(100, 100),
    img_norm=dict(
        mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2533    | 0.1739    | 0.8322     | 0.4926     | 0.8192     | 0.8366     | 0.3559     |

Load model checkpoint from /home/cixie/shaoyuan/BEV-Attack/models/bevdet/bevdet-r101.pth
## Model Configuration

```
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDataset_Adv'
data_root = '../../nuscenes_mini/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        is_train=True,
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                   'transformation_3d_flow', 'img_info'))
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['img_inputs'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    is_train=True,
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    dummy=True,
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.3925, 0.3925],
                    scale_ratio_range=[0.95, 1.05],
                    translation_std=[0, 0, 0],
                    update_img2lidar=True),
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.5,
                    update_img2lidar=True),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                dict(
                    type='ObjectNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='Collect3D',
                    keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'lidar2img', 'depth2img', 'cam2img',
                               'pad_shape', 'scale_factor', 'flip',
                               'pcd_horizontal_flip', 'pcd_vertical_flip',
                               'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                               'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                               'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'img_info'))
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            test_mode=False,
            use_valid_flag=True,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            box_type_3d='LiDAR',
            img_info_prototype='bevdet')),
    val=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'),
    test=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
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
data_config = dict(
    cams=[
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.04)
grid_config = dict(
    xbound=[-51.2, 51.2, 0.8],
    ybound=[-51.2, 51.2, 0.8],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[1.0, 60.0, 1.0])
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 64
model = dict(
    type='BEVDet_Adv',
    img_backbone=dict(
        pretrained='torchvision://resnet101',
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPNForBEVDet',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShoot',
        grid_config=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04),
        use_bev_pool=False,
        numC_Trans=64),
    img_bev_encoder_backbone=dict(type='ResNetForBEVDet', numC_input=64),
    img_bev_encoder_neck=dict(
        type='FPN_LSS', in_channels=640, out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead_Adv',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder_Adv',
            pc_range=[-51.2, -51.2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=9),
        separate_head=dict(
            type='SeparateHead',
            init_bias=-2.19,
            final_kernel=3,
            in_channels=64,
            heads=dict(
                reg=(2, 2),
                height=(1, 2),
                dim=(3, 2),
                rot=(2, 2),
                vel=(2, 2),
                heatmap=(2, 2)),
            num_cls=2),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
        train_cfg=None,
        test_cfg=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    train_cfg=None,
    test_cfg=dict(
        pts=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    pretrained=None)
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
img_norm_cfg = dict(
    mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False)
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=10,
    is_train=True,
    category_specify=False,
    totensor=True,
    dataset_cfg=dict(
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='MultiScaleFlipAug3D',
                    img_scale=(1333, 800),
                    pts_scale_ratio=1,
                    flip=False,
                    transforms=[
                        dict(
                            type='DefaultFormatBundle3D',
                            class_names=[
                                'car', 'truck', 'construction_vehicle', 'bus',
                                'trailer', 'barrier', 'motorcycle', 'bicycle',
                                'pedestrian', 'traffic_cone'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=[
                                'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'
                            ])
                    ])
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            filter_empty_gt=False,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            img_info_prototype='bevdet'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    scale=[0.3],
    max_train_samples=323,
    patch_size=(100, 100),
    img_norm=dict(
        mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

Load model checkpoint from /home/cixie/shaoyuan/BEV-Attack/models/bevdet/bevdet-r101.pth
## Model Configuration

```
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDataset_Adv'
data_root = '../../nuscenes_mini/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        is_train=True,
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                   'transformation_3d_flow', 'img_info'))
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['img_inputs'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    is_train=True,
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    dummy=True,
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.3925, 0.3925],
                    scale_ratio_range=[0.95, 1.05],
                    translation_std=[0, 0, 0],
                    update_img2lidar=True),
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.5,
                    update_img2lidar=True),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                dict(
                    type='ObjectNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='Collect3D',
                    keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'lidar2img', 'depth2img', 'cam2img',
                               'pad_shape', 'scale_factor', 'flip',
                               'pcd_horizontal_flip', 'pcd_vertical_flip',
                               'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                               'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                               'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'img_info'))
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            test_mode=False,
            use_valid_flag=True,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            box_type_3d='LiDAR',
            img_info_prototype='bevdet')),
    val=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'),
    test=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
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
data_config = dict(
    cams=[
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.04)
grid_config = dict(
    xbound=[-51.2, 51.2, 0.8],
    ybound=[-51.2, 51.2, 0.8],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[1.0, 60.0, 1.0])
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 64
model = dict(
    type='BEVDet_Adv',
    img_backbone=dict(
        pretrained='torchvision://resnet101',
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPNForBEVDet',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShoot',
        grid_config=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04),
        use_bev_pool=False,
        numC_Trans=64),
    img_bev_encoder_backbone=dict(type='ResNetForBEVDet', numC_input=64),
    img_bev_encoder_neck=dict(
        type='FPN_LSS', in_channels=640, out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead_Adv',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder_Adv',
            pc_range=[-51.2, -51.2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=9),
        separate_head=dict(
            type='SeparateHead',
            init_bias=-2.19,
            final_kernel=3,
            in_channels=64,
            heads=dict(
                reg=(2, 2),
                height=(1, 2),
                dim=(3, 2),
                rot=(2, 2),
                vel=(2, 2),
                heatmap=(2, 2)),
            num_cls=2),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
        train_cfg=None,
        test_cfg=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    train_cfg=None,
    test_cfg=dict(
        pts=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    pretrained=None)
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
img_norm_cfg = dict(
    mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False)
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=10,
    is_train=True,
    category_specify=False,
    totensor=True,
    dataset_cfg=dict(
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='MultiScaleFlipAug3D',
                    img_scale=(1333, 800),
                    pts_scale_ratio=1,
                    flip=False,
                    transforms=[
                        dict(
                            type='DefaultFormatBundle3D',
                            class_names=[
                                'car', 'truck', 'construction_vehicle', 'bus',
                                'trailer', 'barrier', 'motorcycle', 'bicycle',
                                'pedestrian', 'traffic_cone'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=[
                                'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'
                            ])
                    ])
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            filter_empty_gt=False,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            img_info_prototype='bevdet'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    scale=[0.3],
    max_train_samples=323,
    patch_size=(100, 100),
    img_norm=dict(
        mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

Load model checkpoint from /home/cixie/shaoyuan/BEV-Attack/models/bevdet/bevdet-r101.pth
## Model Configuration

```
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDataset_Adv'
data_root = '../../nuscenes_mini/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        is_train=True,
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                   'transformation_3d_flow', 'img_info'))
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['img_inputs'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    is_train=True,
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    dummy=True,
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.3925, 0.3925],
                    scale_ratio_range=[0.95, 1.05],
                    translation_std=[0, 0, 0],
                    update_img2lidar=True),
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.5,
                    update_img2lidar=True),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                dict(
                    type='ObjectNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='Collect3D',
                    keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'lidar2img', 'depth2img', 'cam2img',
                               'pad_shape', 'scale_factor', 'flip',
                               'pcd_horizontal_flip', 'pcd_vertical_flip',
                               'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                               'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                               'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'img_info'))
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            test_mode=False,
            use_valid_flag=True,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            box_type_3d='LiDAR',
            img_info_prototype='bevdet')),
    val=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'),
    test=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
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
data_config = dict(
    cams=[
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.04)
grid_config = dict(
    xbound=[-51.2, 51.2, 0.8],
    ybound=[-51.2, 51.2, 0.8],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[1.0, 60.0, 1.0])
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 64
model = dict(
    type='BEVDet_Adv',
    img_backbone=dict(
        pretrained='torchvision://resnet101',
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPNForBEVDet',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShoot',
        grid_config=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04),
        use_bev_pool=False,
        numC_Trans=64),
    img_bev_encoder_backbone=dict(type='ResNetForBEVDet', numC_input=64),
    img_bev_encoder_neck=dict(
        type='FPN_LSS', in_channels=640, out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead_Adv',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder_Adv',
            pc_range=[-51.2, -51.2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=9),
        separate_head=dict(
            type='SeparateHead',
            init_bias=-2.19,
            final_kernel=3,
            in_channels=64,
            heads=dict(
                reg=(2, 2),
                height=(1, 2),
                dim=(3, 2),
                rot=(2, 2),
                vel=(2, 2),
                heatmap=(2, 2)),
            num_cls=2),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
        train_cfg=None,
        test_cfg=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    train_cfg=None,
    test_cfg=dict(
        pts=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    pretrained=None)
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
img_norm_cfg = dict(
    mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False)
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=10,
    is_train=True,
    category_specify=False,
    totensor=True,
    dataset_cfg=dict(
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='MultiScaleFlipAug3D',
                    img_scale=(1333, 800),
                    pts_scale_ratio=1,
                    flip=False,
                    transforms=[
                        dict(
                            type='DefaultFormatBundle3D',
                            class_names=[
                                'car', 'truck', 'construction_vehicle', 'bus',
                                'trailer', 'barrier', 'motorcycle', 'bicycle',
                                'pedestrian', 'traffic_cone'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=[
                                'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'
                            ])
                    ])
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            filter_empty_gt=False,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            img_info_prototype='bevdet'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    scale=[0.3],
    max_train_samples=323,
    patch_size=(100, 100),
    img_norm=dict(
        mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2551    | 0.1758    | 0.8267     | 0.4929     | 0.7899     | 0.8496     | 0.3695     |

Load model checkpoint from /home/cixie/shaoyuan/BEV-Attack/models/bevdet/bevdet-r101.pth
## Model Configuration

```
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDataset_Adv'
data_root = '../../nuscenes_mini/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        is_train=True,
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                   'transformation_3d_flow', 'img_info'))
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['img_inputs'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    is_train=True,
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    dummy=True,
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.3925, 0.3925],
                    scale_ratio_range=[0.95, 1.05],
                    translation_std=[0, 0, 0],
                    update_img2lidar=True),
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.5,
                    update_img2lidar=True),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                dict(
                    type='ObjectNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='Collect3D',
                    keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'lidar2img', 'depth2img', 'cam2img',
                               'pad_shape', 'scale_factor', 'flip',
                               'pcd_horizontal_flip', 'pcd_vertical_flip',
                               'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                               'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                               'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'img_info'))
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            test_mode=False,
            use_valid_flag=True,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            box_type_3d='LiDAR',
            img_info_prototype='bevdet')),
    val=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'),
    test=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
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
data_config = dict(
    cams=[
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.04)
grid_config = dict(
    xbound=[-51.2, 51.2, 0.8],
    ybound=[-51.2, 51.2, 0.8],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[1.0, 60.0, 1.0])
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 64
model = dict(
    type='BEVDet_Adv',
    img_backbone=dict(
        pretrained='torchvision://resnet101',
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPNForBEVDet',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShoot',
        grid_config=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04),
        use_bev_pool=False,
        numC_Trans=64),
    img_bev_encoder_backbone=dict(type='ResNetForBEVDet', numC_input=64),
    img_bev_encoder_neck=dict(
        type='FPN_LSS', in_channels=640, out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead_Adv',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder_Adv',
            pc_range=[-51.2, -51.2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=9),
        separate_head=dict(
            type='SeparateHead',
            init_bias=-2.19,
            final_kernel=3,
            in_channels=64,
            heads=dict(
                reg=(2, 2),
                height=(1, 2),
                dim=(3, 2),
                rot=(2, 2),
                vel=(2, 2),
                heatmap=(2, 2)),
            num_cls=2),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
        train_cfg=None,
        test_cfg=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    train_cfg=None,
    test_cfg=dict(
        pts=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    pretrained=None)
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
img_norm_cfg = dict(
    mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False)
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=10,
    is_train=False,
    category_specify=False,
    totensor=True,
    dataset_cfg=dict(
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='MultiScaleFlipAug3D',
                    img_scale=(1333, 800),
                    pts_scale_ratio=1,
                    flip=False,
                    transforms=[
                        dict(
                            type='DefaultFormatBundle3D',
                            class_names=[
                                'car', 'truck', 'construction_vehicle', 'bus',
                                'trailer', 'barrier', 'motorcycle', 'bicycle',
                                'pedestrian', 'traffic_cone'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=[
                                'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'
                            ])
                    ])
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            filter_empty_gt=False,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            img_info_prototype='bevdet'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    scale=[0.3],
    max_train_samples=323,
    patch_size=(100, 100),
    img_norm=dict(
        mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False),
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2670    | 0.1924    | 0.8274     | 0.4888     | 0.7739     | 0.8347     | 0.3671     |

Load model checkpoint from /home/cixie/shaoyuan/BEV-Attack/models/bevdet/bevdet-r101.pth
## Model Configuration

```
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDataset_Adv'
data_root = '../../nuscenes_mini/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        is_train=True,
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                   'transformation_3d_flow', 'img_info'))
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['img_inputs'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    is_train=True,
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    dummy=True,
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.3925, 0.3925],
                    scale_ratio_range=[0.95, 1.05],
                    translation_std=[0, 0, 0],
                    update_img2lidar=True),
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.5,
                    update_img2lidar=True),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                dict(
                    type='ObjectNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='Collect3D',
                    keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'lidar2img', 'depth2img', 'cam2img',
                               'pad_shape', 'scale_factor', 'flip',
                               'pcd_horizontal_flip', 'pcd_vertical_flip',
                               'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                               'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                               'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'img_info'))
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            test_mode=False,
            use_valid_flag=True,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            box_type_3d='LiDAR',
            img_info_prototype='bevdet')),
    val=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'),
    test=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
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
data_config = dict(
    cams=[
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.04)
grid_config = dict(
    xbound=[-51.2, 51.2, 0.8],
    ybound=[-51.2, 51.2, 0.8],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[1.0, 60.0, 1.0])
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 64
model = dict(
    type='BEVDet_Adv',
    img_backbone=dict(
        pretrained='torchvision://resnet101',
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPNForBEVDet',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShoot',
        grid_config=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04),
        use_bev_pool=False,
        numC_Trans=64),
    img_bev_encoder_backbone=dict(type='ResNetForBEVDet', numC_input=64),
    img_bev_encoder_neck=dict(
        type='FPN_LSS', in_channels=640, out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead_Adv',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder_Adv',
            pc_range=[-51.2, -51.2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=9),
        separate_head=dict(
            type='SeparateHead',
            init_bias=-2.19,
            final_kernel=3,
            in_channels=64,
            heads=dict(
                reg=(2, 2),
                height=(1, 2),
                dim=(3, 2),
                rot=(2, 2),
                vel=(2, 2),
                heatmap=(2, 2)),
            num_cls=2),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
        train_cfg=None,
        test_cfg=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    train_cfg=None,
    test_cfg=dict(
        pts=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    pretrained=None)
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
img_norm_cfg = dict(
    mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False)
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=10,
    is_train=False,
    category_specify=False,
    totensor=True,
    dataset_cfg=dict(
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='MultiScaleFlipAug3D',
                    img_scale=(1333, 800),
                    pts_scale_ratio=1,
                    flip=False,
                    transforms=[
                        dict(
                            type='DefaultFormatBundle3D',
                            class_names=[
                                'car', 'truck', 'construction_vehicle', 'bus',
                                'trailer', 'barrier', 'motorcycle', 'bicycle',
                                'pedestrian', 'traffic_cone'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=[
                                'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'
                            ])
                    ])
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            filter_empty_gt=False,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            img_info_prototype='bevdet'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    scale=[0.3],
    max_train_samples=323,
    patch_size=(100, 100),
    img_norm=dict(
        mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False),
    patch_path=[
        '/home/cixie/shaoyuan/BEV-Attack/zoo/BEVDet/uni_patch_new/BEVDepth_Adv_coslr_size100_scale0.3_lr0.0392156862745098_sample323.pkl'
    ],
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2467    | 0.1829    | 0.8271     | 0.4970     | 0.8481     | 0.8978     | 0.3772     |

Load model checkpoint from /home/cixie/shaoyuan/BEV-Attack/models/bevdet/bevdet-r101.pth
## Model Configuration

```
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDataset_Adv'
data_root = '../../nuscenes_mini/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        is_train=True,
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                   'transformation_3d_flow', 'img_info'))
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['img_inputs'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    is_train=True,
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    dummy=True,
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.3925, 0.3925],
                    scale_ratio_range=[0.95, 1.05],
                    translation_std=[0, 0, 0],
                    update_img2lidar=True),
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.5,
                    update_img2lidar=True),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                dict(
                    type='ObjectNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='Collect3D',
                    keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'lidar2img', 'depth2img', 'cam2img',
                               'pad_shape', 'scale_factor', 'flip',
                               'pcd_horizontal_flip', 'pcd_vertical_flip',
                               'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                               'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                               'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'img_info'))
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            test_mode=False,
            use_valid_flag=True,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            box_type_3d='LiDAR',
            img_info_prototype='bevdet')),
    val=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'),
    test=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
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
data_config = dict(
    cams=[
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.04)
grid_config = dict(
    xbound=[-51.2, 51.2, 0.8],
    ybound=[-51.2, 51.2, 0.8],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[1.0, 60.0, 1.0])
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 64
model = dict(
    type='BEVDet_Adv',
    img_backbone=dict(
        pretrained='torchvision://resnet101',
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPNForBEVDet',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShoot',
        grid_config=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04),
        use_bev_pool=False,
        numC_Trans=64),
    img_bev_encoder_backbone=dict(type='ResNetForBEVDet', numC_input=64),
    img_bev_encoder_neck=dict(
        type='FPN_LSS', in_channels=640, out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead_Adv',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder_Adv',
            pc_range=[-51.2, -51.2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=9),
        separate_head=dict(
            type='SeparateHead',
            init_bias=-2.19,
            final_kernel=3,
            in_channels=64,
            heads=dict(
                reg=(2, 2),
                height=(1, 2),
                dim=(3, 2),
                rot=(2, 2),
                vel=(2, 2),
                heatmap=(2, 2)),
            num_cls=2),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
        train_cfg=None,
        test_cfg=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    train_cfg=None,
    test_cfg=dict(
        pts=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    pretrained=None)
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
img_norm_cfg = dict(
    mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False)
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=10,
    is_train=False,
    category_specify=False,
    totensor=True,
    dataset_cfg=dict(
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='MultiScaleFlipAug3D',
                    img_scale=(1333, 800),
                    pts_scale_ratio=1,
                    flip=False,
                    transforms=[
                        dict(
                            type='DefaultFormatBundle3D',
                            class_names=[
                                'car', 'truck', 'construction_vehicle', 'bus',
                                'trailer', 'barrier', 'motorcycle', 'bicycle',
                                'pedestrian', 'traffic_cone'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=[
                                'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'
                            ])
                    ])
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            filter_empty_gt=False,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            img_info_prototype='bevdet'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    scale=[0.3],
    max_train_samples=323,
    patch_size=(100, 100),
    img_norm=dict(
        mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False),
    patch_path=[
        '/home/cixie/shaoyuan/BEV-Attack/mmdet_adv/uni_patch_new/BEVFormer_coslr_size100_scale0.3_lr10_sample323.pkl'
    ],
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2342    | 0.1668    | 0.8448     | 0.4977     | 0.8804     | 0.9025     | 0.3668     |

Load model checkpoint from /home/cixie/shaoyuan/BEV-Attack/models/bevdet/bevdet-r101.pth
## Model Configuration

```
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDataset_Adv'
data_root = '../../nuscenes_mini/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        is_train=True,
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                   'transformation_3d_flow', 'img_info'))
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['img_inputs'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    is_train=True,
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    dummy=True,
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.3925, 0.3925],
                    scale_ratio_range=[0.95, 1.05],
                    translation_std=[0, 0, 0],
                    update_img2lidar=True),
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.5,
                    update_img2lidar=True),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                dict(
                    type='ObjectNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='Collect3D',
                    keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'lidar2img', 'depth2img', 'cam2img',
                               'pad_shape', 'scale_factor', 'flip',
                               'pcd_horizontal_flip', 'pcd_vertical_flip',
                               'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                               'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                               'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'img_info'))
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            test_mode=False,
            use_valid_flag=True,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            box_type_3d='LiDAR',
            img_info_prototype='bevdet')),
    val=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'),
    test=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
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
data_config = dict(
    cams=[
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.04)
grid_config = dict(
    xbound=[-51.2, 51.2, 0.8],
    ybound=[-51.2, 51.2, 0.8],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[1.0, 60.0, 1.0])
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 64
model = dict(
    type='BEVDet_Adv',
    img_backbone=dict(
        pretrained='torchvision://resnet101',
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPNForBEVDet',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShoot',
        grid_config=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04),
        use_bev_pool=False,
        numC_Trans=64),
    img_bev_encoder_backbone=dict(type='ResNetForBEVDet', numC_input=64),
    img_bev_encoder_neck=dict(
        type='FPN_LSS', in_channels=640, out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead_Adv',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder_Adv',
            pc_range=[-51.2, -51.2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=9),
        separate_head=dict(
            type='SeparateHead',
            init_bias=-2.19,
            final_kernel=3,
            in_channels=64,
            heads=dict(
                reg=(2, 2),
                height=(1, 2),
                dim=(3, 2),
                rot=(2, 2),
                vel=(2, 2),
                heatmap=(2, 2)),
            num_cls=2),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
        train_cfg=None,
        test_cfg=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    train_cfg=None,
    test_cfg=dict(
        pts=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    pretrained=None)
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
img_norm_cfg = dict(
    mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False)
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=10,
    is_train=False,
    category_specify=False,
    totensor=True,
    dataset_cfg=dict(
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='MultiScaleFlipAug3D',
                    img_scale=(1333, 800),
                    pts_scale_ratio=1,
                    flip=False,
                    transforms=[
                        dict(
                            type='DefaultFormatBundle3D',
                            class_names=[
                                'car', 'truck', 'construction_vehicle', 'bus',
                                'trailer', 'barrier', 'motorcycle', 'bicycle',
                                'pedestrian', 'traffic_cone'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=[
                                'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'
                            ])
                    ])
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            filter_empty_gt=False,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            img_info_prototype='bevdet'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    scale=[0.3],
    max_train_samples=323,
    patch_size=(100, 100),
    img_norm=dict(
        mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False),
    patch_path=[
        '/home/cixie/shaoyuan/BEV-Attack/mmdet_adv/uni_patch_new/Detr3D_coslr_size100_scale0.3_lr10_sample323.pkl'
    ],
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2335    | 0.1693    | 0.8582     | 0.4997     | 0.8622     | 0.9020     | 0.3899     |

Load model checkpoint from /home/cixie/shaoyuan/BEV-Attack/models/bevdet/bevdet-r101.pth
## Model Configuration

```
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDataset_Adv'
data_root = '../../nuscenes_mini/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        is_train=True,
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                   'transformation_3d_flow', 'img_info'))
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['img_inputs'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    is_train=True,
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    dummy=True,
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.3925, 0.3925],
                    scale_ratio_range=[0.95, 1.05],
                    translation_std=[0, 0, 0],
                    update_img2lidar=True),
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.5,
                    update_img2lidar=True),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                dict(
                    type='ObjectNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='Collect3D',
                    keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'lidar2img', 'depth2img', 'cam2img',
                               'pad_shape', 'scale_factor', 'flip',
                               'pcd_horizontal_flip', 'pcd_vertical_flip',
                               'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                               'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                               'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'img_info'))
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            test_mode=False,
            use_valid_flag=True,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            box_type_3d='LiDAR',
            img_info_prototype='bevdet')),
    val=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'),
    test=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
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
data_config = dict(
    cams=[
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.04)
grid_config = dict(
    xbound=[-51.2, 51.2, 0.8],
    ybound=[-51.2, 51.2, 0.8],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[1.0, 60.0, 1.0])
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 64
model = dict(
    type='BEVDet_Adv',
    img_backbone=dict(
        pretrained='torchvision://resnet101',
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPNForBEVDet',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShoot',
        grid_config=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04),
        use_bev_pool=False,
        numC_Trans=64),
    img_bev_encoder_backbone=dict(type='ResNetForBEVDet', numC_input=64),
    img_bev_encoder_neck=dict(
        type='FPN_LSS', in_channels=640, out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead_Adv',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder_Adv',
            pc_range=[-51.2, -51.2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=9),
        separate_head=dict(
            type='SeparateHead',
            init_bias=-2.19,
            final_kernel=3,
            in_channels=64,
            heads=dict(
                reg=(2, 2),
                height=(1, 2),
                dim=(3, 2),
                rot=(2, 2),
                vel=(2, 2),
                heatmap=(2, 2)),
            num_cls=2),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
        train_cfg=None,
        test_cfg=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    train_cfg=None,
    test_cfg=dict(
        pts=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    pretrained=None)
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
img_norm_cfg = dict(
    mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False)
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=10,
    is_train=False,
    category_specify=False,
    totensor=True,
    dataset_cfg=dict(
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='MultiScaleFlipAug3D',
                    img_scale=(1333, 800),
                    pts_scale_ratio=1,
                    flip=False,
                    transforms=[
                        dict(
                            type='DefaultFormatBundle3D',
                            class_names=[
                                'car', 'truck', 'construction_vehicle', 'bus',
                                'trailer', 'barrier', 'motorcycle', 'bicycle',
                                'pedestrian', 'traffic_cone'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=[
                                'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'
                            ])
                    ])
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            filter_empty_gt=False,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            img_info_prototype='bevdet'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    scale=[0.3],
    max_train_samples=323,
    patch_size=(100, 100),
    img_norm=dict(
        mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False),
    patch_path=[
        '/home/cixie/shaoyuan/BEV-Attack/mmdet_adv/uni_patch_new/Petr3D_Adv_coslr_size100_scale0.3_lr10_sample323.pkl'
    ],
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1398    | 0.0941    | 0.8850     | 0.6259     | 0.9259     | 1.1241     | 0.6358     |

Load model checkpoint from /home/cixie/shaoyuan/BEV-Attack/models/bevdet/bevdet-r101.pth
## Model Configuration

```
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDataset_Adv'
data_root = '../../nuscenes_mini/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        is_train=True,
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                   'transformation_3d_flow', 'img_info'))
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['img_inputs'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    is_train=True,
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    dummy=True,
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.3925, 0.3925],
                    scale_ratio_range=[0.95, 1.05],
                    translation_std=[0, 0, 0],
                    update_img2lidar=True),
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.5,
                    update_img2lidar=True),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                dict(
                    type='ObjectNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='Collect3D',
                    keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'lidar2img', 'depth2img', 'cam2img',
                               'pad_shape', 'scale_factor', 'flip',
                               'pcd_horizontal_flip', 'pcd_vertical_flip',
                               'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                               'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                               'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'img_info'))
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            test_mode=False,
            use_valid_flag=True,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            box_type_3d='LiDAR',
            img_info_prototype='bevdet')),
    val=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'),
    test=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
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
data_config = dict(
    cams=[
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.04)
grid_config = dict(
    xbound=[-51.2, 51.2, 0.8],
    ybound=[-51.2, 51.2, 0.8],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[1.0, 60.0, 1.0])
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 64
model = dict(
    type='BEVDet_Adv',
    img_backbone=dict(
        pretrained='torchvision://resnet101',
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPNForBEVDet',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShoot',
        grid_config=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04),
        use_bev_pool=False,
        numC_Trans=64),
    img_bev_encoder_backbone=dict(type='ResNetForBEVDet', numC_input=64),
    img_bev_encoder_neck=dict(
        type='FPN_LSS', in_channels=640, out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead_Adv',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder_Adv',
            pc_range=[-51.2, -51.2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=9),
        separate_head=dict(
            type='SeparateHead',
            init_bias=-2.19,
            final_kernel=3,
            in_channels=64,
            heads=dict(
                reg=(2, 2),
                height=(1, 2),
                dim=(3, 2),
                rot=(2, 2),
                vel=(2, 2),
                heatmap=(2, 2)),
            num_cls=2),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
        train_cfg=None,
        test_cfg=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    train_cfg=None,
    test_cfg=dict(
        pts=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    pretrained=None)
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
img_norm_cfg = dict(
    mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False)
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=10,
    is_train=False,
    category_specify=False,
    totensor=True,
    dataset_cfg=dict(
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='MultiScaleFlipAug3D',
                    img_scale=(1333, 800),
                    pts_scale_ratio=1,
                    flip=False,
                    transforms=[
                        dict(
                            type='DefaultFormatBundle3D',
                            class_names=[
                                'car', 'truck', 'construction_vehicle', 'bus',
                                'trailer', 'barrier', 'motorcycle', 'bicycle',
                                'pedestrian', 'traffic_cone'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=[
                                'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'
                            ])
                    ])
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            filter_empty_gt=False,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            img_info_prototype='bevdet'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    scale=[0.3],
    max_train_samples=323,
    patch_size=(100, 100),
    img_norm=dict(
        mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False),
    patch_path=[
        '/home/cixie/shaoyuan/BEV-Attack/mmdet_adv/uni_patch_new/FCOSMono3D_coslr_size100_scale0.3_lr10_sample1938.pkl'
    ],
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1461    | 0.1018    | 0.8883     | 0.6197     | 0.8943     | 1.1006     | 0.6451     |

Load model checkpoint from /home/cixie/shaoyuan/BEV-Attack/models/bevdet/bevdet-r101.pth
## Model Configuration

```
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDataset_Adv'
data_root = '../../nuscenes_mini/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        is_train=True,
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        dummy=True,
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                   'transformation_3d_flow', 'img_info'))
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04)),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['img_inputs'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=False,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    is_train=True,
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    dummy=True,
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.3925, 0.3925],
                    scale_ratio_range=[0.95, 1.05],
                    translation_std=[0, 0, 0],
                    update_img2lidar=True),
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.5,
                    update_img2lidar=True),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
                dict(
                    type='ObjectNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='Collect3D',
                    keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'lidar2img', 'depth2img', 'cam2img',
                               'pad_shape', 'scale_factor', 'flip',
                               'pcd_horizontal_flip', 'pcd_vertical_flip',
                               'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                               'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                               'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'img_info'))
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            test_mode=False,
            use_valid_flag=True,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            box_type_3d='LiDAR',
            img_info_prototype='bevdet')),
    val=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'),
    test=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file='../../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.04)),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        img_info_prototype='bevdet'))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
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
data_config = dict(
    cams=[
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.04)
grid_config = dict(
    xbound=[-51.2, 51.2, 0.8],
    ybound=[-51.2, 51.2, 0.8],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[1.0, 60.0, 1.0])
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 64
model = dict(
    type='BEVDet_Adv',
    img_backbone=dict(
        pretrained='torchvision://resnet101',
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPNForBEVDet',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShoot',
        grid_config=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0]),
        data_config=dict(
            cams=[
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.04),
        use_bev_pool=False,
        numC_Trans=64),
    img_bev_encoder_backbone=dict(type='ResNetForBEVDet', numC_input=64),
    img_bev_encoder_neck=dict(
        type='FPN_LSS', in_channels=640, out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead_Adv',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder_Adv',
            pc_range=[-51.2, -51.2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=9),
        separate_head=dict(
            type='SeparateHead',
            init_bias=-2.19,
            final_kernel=3,
            in_channels=64,
            heads=dict(
                reg=(2, 2),
                height=(1, 2),
                dim=(3, 2),
                rot=(2, 2),
                vel=(2, 2),
                heatmap=(2, 2)),
            num_cls=2),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
        train_cfg=None,
        test_cfg=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    train_cfg=None,
    test_cfg=dict(
        pts=dict(
            pc_range=[-51.2, -51.2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])),
    pretrained=None)
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
img_norm_cfg = dict(
    mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False)
attack_severity_type = 'scale'
attack = dict(
    type='UniversalPatchAttackOptim',
    epoch=1,
    lr=10,
    is_train=False,
    category_specify=False,
    totensor=True,
    dataset_cfg=dict(
        dataset=dict(
            type='NuScenesDataset_Adv',
            data_root='../../nuscenes_mini/',
            ann_file='../../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(
                    type='LoadMultiViewImageFromFiles_BEVDet',
                    data_config=dict(
                        cams=[
                            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.04)),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='MultiScaleFlipAug3D',
                    img_scale=(1333, 800),
                    pts_scale_ratio=1,
                    flip=False,
                    transforms=[
                        dict(
                            type='DefaultFormatBundle3D',
                            class_names=[
                                'car', 'truck', 'construction_vehicle', 'bus',
                                'trailer', 'barrier', 'motorcycle', 'bicycle',
                                'pedestrian', 'traffic_cone'
                            ],
                            with_label=False),
                        dict(
                            type='Collect3D',
                            keys=[
                                'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'
                            ])
                    ])
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            filter_empty_gt=False,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            img_info_prototype='bevdet'),
        shuffle=True,
        workers_per_gpu=32),
    dynamic_patch_size=True,
    scale=[0.3],
    max_train_samples=323,
    patch_size=(100, 100),
    img_norm=dict(
        mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False),
    patch_path=[
        '/home/cixie/shaoyuan/BEV-Attack/mmdet_adv/uni_patch_new/PGDMono3D_coslr_size100_scale0.3_lr10_sample1938.pkl'
    ],
    loss_fn=dict(
        type='TargetedClassificationObjective',
        num_cls=10,
        random=True,
        thresh=0.1),
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### scale 0.3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1552    | 0.1186    | 0.8570     | 0.6406     | 0.9110     | 1.0092     | 0.6331     |

