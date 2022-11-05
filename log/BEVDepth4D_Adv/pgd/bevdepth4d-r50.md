Load model checkpoint from /home/cixie/shaoyuan/BEV-Attack/models/bevdepth/bevdepth4d-r50.pth
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
            resize_test=0.04),
        sequential=True,
        aligned=True,
        trans_only=False),
    dict(
        type='LoadPointsFromFile',
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
        type='PointToMultiViewDepth',
        grid_config=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0])),
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
            resize_test=0.04),
        sequential=True,
        aligned=True,
        trans_only=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='PointToMultiViewDepth',
        grid_config=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0])),
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
                keys=['gt_bboxes_3d', 'gt_labels_3d', 'img_inputs'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                           'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
                           'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                           'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                           'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                           'pcd_rotation', 'pts_filename',
                           'transformation_3d_flow', 'adjacent',
                           'adjacent_type'))
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
            resize_test=0.04),
        sequential=True,
        aligned=True,
        trans_only=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='PointToMultiViewDepth',
        grid_config=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[1.0, 60.0, 1.0])),
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
    workers_per_gpu=4,
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
            ann_file=
            '../../nuscenes_mini/nuscenes_infos_train_4d_interval3_max60.pkl',
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
                        resize_test=0.04),
                    sequential=True,
                    aligned=True,
                    trans_only=False),
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
                    type='PointToMultiViewDepth',
                    grid_config=dict(
                        xbound=[-51.2, 51.2, 0.8],
                        ybound=[-51.2, 51.2, 0.8],
                        zbound=[-10.0, 10.0, 20.0],
                        dbound=[1.0, 60.0, 1.0])),
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
            img_info_prototype='bevdet_sequential',
            speed_mode='abs_dis',
            max_interval=9,
            min_interval=2,
            prev_only=True,
            fix_direction=True)),
    val=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file=
        '../../nuscenes_mini/nuscenes_infos_val_4d_interval3_max60.pkl',
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
                    resize_test=0.04),
                sequential=True,
                aligned=True,
                trans_only=False),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='PointToMultiViewDepth',
                grid_config=dict(
                    xbound=[-51.2, 51.2, 0.8],
                    ybound=[-51.2, 51.2, 0.8],
                    zbound=[-10.0, 10.0, 20.0],
                    dbound=[1.0, 60.0, 1.0])),
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
                        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img_inputs'],
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'lidar2img', 'depth2img', 'cam2img',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'pcd_horizontal_flip', 'pcd_vertical_flip',
                                   'box_mode_3d', 'box_type_3d',
                                   'img_norm_cfg', 'pcd_trans', 'sample_idx',
                                   'pcd_scale_factor', 'pcd_rotation',
                                   'pts_filename', 'transformation_3d_flow',
                                   'adjacent', 'adjacent_type'))
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
        img_info_prototype='bevdet_sequential',
        max_interval=10,
        fix_direction=True),
    test=dict(
        type='NuScenesDataset_Adv',
        data_root='../../nuscenes_mini/',
        ann_file=
        '../../nuscenes_mini/nuscenes_infos_val_4d_interval3_max60.pkl',
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
                    resize_test=0.04),
                sequential=True,
                aligned=True,
                trans_only=False),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='PointToMultiViewDepth',
                grid_config=dict(
                    xbound=[-51.2, 51.2, 0.8],
                    ybound=[-51.2, 51.2, 0.8],
                    zbound=[-10.0, 10.0, 20.0],
                    dbound=[1.0, 60.0, 1.0])),
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
                        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img_inputs'],
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'lidar2img', 'depth2img', 'cam2img',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'pcd_horizontal_flip', 'pcd_vertical_flip',
                                   'box_mode_3d', 'box_type_3d',
                                   'img_norm_cfg', 'pcd_trans', 'sample_idx',
                                   'pcd_scale_factor', 'pcd_rotation',
                                   'pts_filename', 'transformation_3d_flow',
                                   'adjacent', 'adjacent_type'))
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
        img_info_prototype='bevdet_sequential',
        speed_mode='abs_dis',
        max_interval=10,
        fix_direction=True))
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
    type='BEVDepth4D_Adv',
    aligned=True,
    detach=True,
    before=True,
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
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
        type='ViewTransformerLSSBEVDepth',
        loss_depth_weight=100.0,
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
        numC_Trans=64,
        extra_depth_net=dict(
            type='ResNetForBEVDet',
            numC_input=256,
            num_layer=[3],
            num_channels=[256],
            stride=[1])),
    img_bev_encoder_backbone=dict(
        type='ResNetForBEVDet', numC_input=128, num_channels=[128, 256, 512]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS', in_channels=640, out_channels=256),
    pre_process=dict(
        type='ResNetForBEVDet',
        numC_input=64,
        num_layer=[2],
        num_channels=[64],
        stride=[1],
        backbone_output_ids=[0]),
    pts_bbox_head=dict(
        type='CenterHead_Adv',
        task_specific=True,
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
attack_severity_type = 'num_steps'
attack = dict(
    type='PGD',
    epsilon=[0.08562376915831835, 0.08753501400560224, 0.08714596949891067],
    step_size=[
        0.001712475383166367, 0.0017507002801120449, 0.0017429193899782137
    ],
    num_steps=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50],
    img_norm=dict(
        mean=[[0.485, 0.456, 0.406]], std=[0.229, 0.224, 0.225], to_rgb=False),
    single_camera=False,
    totensor=True,
    sequential=True,
    loss_fn=dict(type='ClassficationObjective', activate=False),
    category='Madry',
    rand_init=True,
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### num_steps 1

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3826    | 0.2941    | 0.7593     | 0.4671     | 0.6462     | 0.4844     | 0.2875     |

### num_steps 2

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3586    | 0.2388    | 0.7356     | 0.4636     | 0.6287     | 0.4908     | 0.2889     |

### num_steps 3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3250    | 0.1804    | 0.8014     | 0.4708     | 0.6133     | 0.4707     | 0.2956     |

### num_steps 4

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3006    | 0.1519    | 0.8821     | 0.4762     | 0.6169     | 0.4909     | 0.2868     |

### num_steps 5

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2429    | 0.1191    | 0.8609     | 0.5478     | 0.7380     | 0.6080     | 0.4117     |

### num_steps 6

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2674    | 0.0987    | 0.8529     | 0.4841     | 0.6477     | 0.5335     | 0.3013     |

### num_steps 7

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2250    | 0.0860    | 0.8592     | 0.5523     | 0.7420     | 0.6164     | 0.4101     |

### num_steps 8

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2166    | 0.0693    | 0.8692     | 0.5545     | 0.7269     | 0.6157     | 0.4141     |

### num_steps 9

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1758    | 0.0560    | 0.8875     | 0.6293     | 0.7052     | 0.7668     | 0.5330     |

### num_steps 10

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1998    | 0.0502    | 0.8676     | 0.5531     | 0.7533     | 0.6675     | 0.4119     |

### num_steps 20

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1452    | 0.0104    | 0.8491     | 0.6506     | 0.7329     | 0.8302     | 0.5372     |

### num_steps 30

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.0729    | 0.0021    | 0.9465     | 0.8137     | 0.8356     | 0.8878     | 0.7984     |

### num_steps 40

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.0562    | 0.0003    | 1.0052     | 0.8275     | 0.8278     | 1.0106     | 0.7837     |

### num_steps 50

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.0319    | 0.0000    | 0.9888     | 0.9202     | 0.9237     | 0.9442     | 0.9038     |

