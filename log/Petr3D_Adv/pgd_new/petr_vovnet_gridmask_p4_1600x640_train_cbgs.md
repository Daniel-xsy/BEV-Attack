Load model checkpoint from /home/cixie/shaoyuan/BEV-Attack/models/petr/petr-vov-p4-1600x640.pth
## Model Configuration

```
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'PETRCustomNuScenesDataset'
data_root = '../nuscenes_mini/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
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
        type='ResizeCropFlipImage',
        data_aug_conf=dict(
            resize_lim=(0.94, 1.25),
            final_dim=(640, 1600),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=True),
        training=True),
    dict(
        type='GlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        reverse_angle=True,
        training=True),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='ResizeCropFlipImage',
        data_aug_conf=dict(
            resize_lim=(0.94, 1.25),
            final_dim=(640, 1600),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=True),
        training=False),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
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
                type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
        ])
]
eval_pipeline = [
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
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=32,
    train=dict(
        type='CBGSDataset',
        data_root='../nuscenes_mini/',
        ann_file='../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
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
            type='PETRCustomNuScenesDataset',
            data_root='../nuscenes_mini/',
            ann_file='../nuscenes_mini/nuscenes_infos_temporal_train.pkl',
            pipeline=[
                dict(type='LoadMultiViewImageFromFiles', to_float32=True),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    with_attr_label=False),
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
                    type='ResizeCropFlipImage',
                    data_aug_conf=dict(
                        resize_lim=(0.94, 1.25),
                        final_dim=(640, 1600),
                        bot_pct_lim=(0.0, 0.0),
                        rot_lim=(0.0, 0.0),
                        H=900,
                        W=1600,
                        rand_flip=True),
                    training=True),
                dict(
                    type='GlobalRotScaleTransImage',
                    rot_range=[-0.3925, 0.3925],
                    translation_std=[0, 0, 0],
                    scale_ratio_range=[0.95, 1.05],
                    reverse_angle=True,
                    training=True),
                dict(
                    type='NormalizeMultiviewImage',
                    mean=[103.53, 116.28, 123.675],
                    std=[57.375, 57.12, 58.395],
                    to_rgb=False),
                dict(type='PadMultiViewImage', size_divisor=32),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='Collect3D',
                    keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=True),
            test_mode=False,
            use_valid_flag=True,
            box_type_3d='LiDAR')),
    val=dict(
        type='PETRCustomNuScenesDataset',
        data_root='../nuscenes_mini/',
        ann_file='../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=False),
            dict(
                type='ResizeCropFlipImage',
                data_aug_conf=dict(
                    resize_lim=(0.94, 1.25),
                    final_dim=(640, 1600),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=False),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
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
                        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
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
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False),
    test=dict(
        type='PETRCustomNuScenesDataset',
        data_root='../nuscenes_mini/',
        ann_file='../nuscenes_mini/nuscenes_infos_temporal_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=False),
            dict(
                type='ResizeCropFlipImage',
                data_aug_conf=dict(
                    resize_lim=(0.94, 1.25),
                    final_dim=(640, 1600),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=False),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
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
                        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
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
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=True,
            with_label_3d=True,
            with_attr_label=False),
        dict(
            type='ResizeCropFlipImage',
            data_aug_conf=dict(
                resize_lim=(0.94, 1.25),
                final_dim=(640, 1600),
                bot_pct_lim=(0.0, 0.0),
                rot_lim=(0.0, 0.0),
                H=900,
                W=1600,
                rand_flip=True),
            training=False),
        dict(
            type='NormalizeMultiviewImage',
            mean=[103.53, 116.28, 123.675],
            std=[57.375, 57.12, 58.395],
            to_rgb=False),
        dict(type='PadMultiViewImage', size_divisor=32),
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
                    keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
            ])
    ])
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = 'ckpts/fcos3d_vovnet_imgbackbone-remapped.pth'
resume_from = None
workflow = [('train', 1)]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin = True
plugin_dir = ['projects/mmdet3d_plugin/']
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False)
model = dict(
    type='Petr3D_Adv',
    use_grid_mask=True,
    img_backbone=dict(
        type='VoVNetCP',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage4', 'stage5')),
    img_neck=dict(
        type='CPFPN', in_channels=[768, 1024], out_channels=256, num_outs=2),
    pts_bbox_head=dict(
        type='PETRHead_Adv',
        num_classes=10,
        in_channels=256,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        normedlinear=False,
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder_Adv',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            voxel_size=[0.2, 0.2, 8],
            num_classes=10),
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        train_cfg=None,
        test_cfg=None),
    train_cfg=None,
    pretrained=None)
db_sampler = dict(
    data_root='../nuscenes_mini/',
    info_path='../nuscenes_mini/nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=dict(backend='disk')))
ida_aug_conf = dict(
    resize_lim=(0.94, 1.25),
    final_dim=(640, 1600),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900,
    W=1600,
    rand_flip=True)
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale=512.0,
    grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
total_epochs = 24
find_unused_parameters = False
runner = dict(type='EpochBasedRunner', max_epochs=24)
attack_severity_type = 'num_steps'
attack = dict(
    type='PGD',
    epsilon=[0.08714596949891068, 0.08753501400560225, 0.08562376915831835],
    step_size=[
        0.0017429193899782137, 0.0017507002801120449, 0.0017124753831663669
    ],
    num_steps=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50],
    img_norm=dict(
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        to_rgb=False),
    single_camera=False,
    loss_fn=dict(type='ClassficationObjective', activate=False),
    category='Madry',
    rand_init=True,
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))

```

### num_steps 1

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3576    | 0.2983    | 0.7335     | 0.4706     | 0.6626     | 0.7582     | 0.2903     |

### num_steps 2

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3018    | 0.2264    | 0.7748     | 0.4715     | 0.7296     | 0.8439     | 0.2945     |

### num_steps 3

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2598    | 0.1660    | 0.8027     | 0.4814     | 0.7924     | 0.8624     | 0.2929     |

### num_steps 4

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2487    | 0.1316    | 0.8174     | 0.4864     | 0.7333     | 0.8318     | 0.3022     |

### num_steps 5

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2110    | 0.1117    | 0.8167     | 0.4955     | 0.8425     | 0.9791     | 0.3147     |

### num_steps 6

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1990    | 0.0864    | 0.8073     | 0.4900     | 0.8114     | 1.1215     | 0.3331     |

### num_steps 7

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1827    | 0.0748    | 0.8564     | 0.4969     | 0.8499     | 1.1298     | 0.3440     |

### num_steps 8

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1688    | 0.0558    | 0.9103     | 0.5014     | 0.8363     | 1.2022     | 0.3426     |

### num_steps 9

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1612    | 0.0464    | 0.8550     | 0.5055     | 0.9300     | 1.4390     | 0.3295     |

### num_steps 10

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1562    | 0.0383    | 0.9138     | 0.5288     | 0.8224     | 1.4500     | 0.3645     |

### num_steps 20

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.0817    | 0.0016    | 0.8892     | 0.6846     | 1.0240     | 1.6567     | 0.6174     |

### num_steps 30

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.0558    | 0.0000    | 0.9709     | 0.7594     | 1.0053     | 1.3710     | 0.7118     |

### num_steps 40

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.0073    | 0.0000    | 0.9623     | 0.9651     | 1.0000     | 1.0000     | 1.0000     |

### num_steps 50

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
{'pts_bbox_NuScenes/car_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/car_AP_dist_1.0': 0.0, 'pts_bbox_NuScenes/car_AP_dist_2.0': 0.0, 'pts_bbox_NuScenes/car_AP_dist_4.0': 0.0, 'pts_bbox_NuScenes/car_trans_err': 1.0, 'pts_bbox_NuScenes/car_scale_err': 1.0, 'pts_bbox_NuScenes/car_orient_err': 1.0, 'pts_bbox_NuScenes/car_vel_err': 1.0, 'pts_bbox_NuScenes/car_attr_err': 1.0, 'pts_bbox_NuScenes/mATE': 1.0, 'pts_bbox_NuScenes/mASE': 1.0, 'pts_bbox_NuScenes/mAOE': 1.0, 'pts_bbox_NuScenes/mAVE': 1.0, 'pts_bbox_NuScenes/mAAE': 1.0, 'pts_bbox_NuScenes/truck_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/truck_AP_dist_1.0': 0.0, 'pts_bbox_NuScenes/truck_AP_dist_2.0': 0.0, 'pts_bbox_NuScenes/truck_AP_dist_4.0': 0.0, 'pts_bbox_NuScenes/truck_trans_err': 1.0, 'pts_bbox_NuScenes/truck_scale_err': 1.0, 'pts_bbox_NuScenes/truck_orient_err': 1.0, 'pts_bbox_NuScenes/truck_vel_err': 1.0, 'pts_bbox_NuScenes/truck_attr_err': 1.0, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_1.0': 0.0, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_2.0': 0.0, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_4.0': 0.0, 'pts_bbox_NuScenes/construction_vehicle_trans_err': 1.0, 'pts_bbox_NuScenes/construction_vehicle_scale_err': 1.0, 'pts_bbox_NuScenes/construction_vehicle_orient_err': 1.0, 'pts_bbox_NuScenes/construction_vehicle_vel_err': 1.0, 'pts_bbox_NuScenes/construction_vehicle_attr_err': 1.0, 'pts_bbox_NuScenes/bus_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/bus_AP_dist_1.0': 0.0, 'pts_bbox_NuScenes/bus_AP_dist_2.0': 0.0, 'pts_bbox_NuScenes/bus_AP_dist_4.0': 0.0, 'pts_bbox_NuScenes/bus_trans_err': 1.0, 'pts_bbox_NuScenes/bus_scale_err': 1.0, 'pts_bbox_NuScenes/bus_orient_err': 1.0, 'pts_bbox_NuScenes/bus_vel_err': 1.0, 'pts_bbox_NuScenes/bus_attr_err': 1.0, 'pts_bbox_NuScenes/trailer_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/trailer_AP_dist_1.0': 0.0, 'pts_bbox_NuScenes/trailer_AP_dist_2.0': 0.0, 'pts_bbox_NuScenes/trailer_AP_dist_4.0': 0.0, 'pts_bbox_NuScenes/trailer_trans_err': 1.0, 'pts_bbox_NuScenes/trailer_scale_err': 1.0, 'pts_bbox_NuScenes/trailer_orient_err': 1.0, 'pts_bbox_NuScenes/trailer_vel_err': 1.0, 'pts_bbox_NuScenes/trailer_attr_err': 1.0, 'pts_bbox_NuScenes/barrier_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/barrier_AP_dist_1.0': 0.0, 'pts_bbox_NuScenes/barrier_AP_dist_2.0': 0.0, 'pts_bbox_NuScenes/barrier_AP_dist_4.0': 0.0, 'pts_bbox_NuScenes/barrier_trans_err': 1.0, 'pts_bbox_NuScenes/barrier_scale_err': 1.0, 'pts_bbox_NuScenes/barrier_orient_err': 1.0, 'pts_bbox_NuScenes/barrier_vel_err': nan, 'pts_bbox_NuScenes/barrier_attr_err': nan, 'pts_bbox_NuScenes/motorcycle_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/motorcycle_AP_dist_1.0': 0.0, 'pts_bbox_NuScenes/motorcycle_AP_dist_2.0': 0.0, 'pts_bbox_NuScenes/motorcycle_AP_dist_4.0': 0.0, 'pts_bbox_NuScenes/motorcycle_trans_err': 1.0, 'pts_bbox_NuScenes/motorcycle_scale_err': 1.0, 'pts_bbox_NuScenes/motorcycle_orient_err': 1.0, 'pts_bbox_NuScenes/motorcycle_vel_err': 1.0, 'pts_bbox_NuScenes/motorcycle_attr_err': 1.0, 'pts_bbox_NuScenes/bicycle_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/bicycle_AP_dist_1.0': 0.0, 'pts_bbox_NuScenes/bicycle_AP_dist_2.0': 0.0, 'pts_bbox_NuScenes/bicycle_AP_dist_4.0': 0.0, 'pts_bbox_NuScenes/bicycle_trans_err': 1.0, 'pts_bbox_NuScenes/bicycle_scale_err': 1.0, 'pts_bbox_NuScenes/bicycle_orient_err': 1.0, 'pts_bbox_NuScenes/bicycle_vel_err': 1.0, 'pts_bbox_NuScenes/bicycle_attr_err': 1.0, 'pts_bbox_NuScenes/pedestrian_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/pedestrian_AP_dist_1.0': 0.0, 'pts_bbox_NuScenes/pedestrian_AP_dist_2.0': 0.0, 'pts_bbox_NuScenes/pedestrian_AP_dist_4.0': 0.0, 'pts_bbox_NuScenes/pedestrian_trans_err': 1.0, 'pts_bbox_NuScenes/pedestrian_scale_err': 1.0, 'pts_bbox_NuScenes/pedestrian_orient_err': 1.0, 'pts_bbox_NuScenes/pedestrian_vel_err': 1.0, 'pts_bbox_NuScenes/pedestrian_attr_err': 1.0, 'pts_bbox_NuScenes/traffic_cone_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/traffic_cone_AP_dist_1.0': 0.0, 'pts_bbox_NuScenes/traffic_cone_AP_dist_2.0': 0.0, 'pts_bbox_NuScenes/traffic_cone_AP_dist_4.0': 0.0, 'pts_bbox_NuScenes/traffic_cone_trans_err': 1.0, 'pts_bbox_NuScenes/traffic_cone_scale_err': 1.0, 'pts_bbox_NuScenes/traffic_cone_orient_err': nan, 'pts_bbox_NuScenes/traffic_cone_vel_err': nan, 'pts_bbox_NuScenes/traffic_cone_attr_err': nan, 'pts_bbox_NuScenes/NDS': 0.0, 'pts_bbox_NuScenes/mAP': 0.0}
