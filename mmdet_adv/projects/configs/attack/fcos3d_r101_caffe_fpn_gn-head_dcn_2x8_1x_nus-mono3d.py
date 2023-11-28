_base_ = [
    '../_base_/datasets/nus-mono3d.py', '../_base_/models/fcos3d.py',
    '../_base_/schedules/mmdet_schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='CustomFCOSMono3D',
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    bbox_head=dict(
        # type='FCOSMono3DHead',
        type='CustomFCOSMono3DHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        use_direction_classifier=True,
        diff_rad_by_sin=True,
        pred_attrs=True,
        pred_velo=True,
        dir_offset=0.7854,  # pi/4
        strides=[8, 16, 32, 64, 128],
        group_reg_dims=(2, 1, 3, 1, 2),  # offset, depth, size, rot, velo
        cls_branch=(256, ),
        reg_branch=(
            (256, ),  # offset
            (256, ),  # depth
            (256, ),  # size
            (256, ),  # rot
            ()  # velo
        ),
        dir_branch=(256, ),
        attr_branch=(256, ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
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
        dcn_on_last_conv=True),
    )

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
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
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'attr_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers2d', 'depths'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='LoadImageInfo3D', with_lidar2img=True, with_sensor2lidar=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='RandomFlip3D'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                                   meta_keys=['filename', 'ori_shape', 'img_shape', 'lidar2img',
                                            'depth2img', 'cam2img', 'pad_shape',
                                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                                            'transformation_3d_flow', 'sensor2lidar_translation',
                                            'sensor2lidar_rotation']),
        ])
]
version = 'v1.0-mini'
dataset_type = 'CustomNuScenesMonoDataset_Adv'
data_root = '../data/nuscenes/'
# dataset_type = 'NuScenesMonoDataset'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(type=dataset_type, 
               data_root=data_root,
               pipeline=train_pipeline, 
               version=version),
    val=dict(type=dataset_type, 
             data_root=data_root,
             pipeline=test_pipeline, 
             version=version, 
             test_mode=False),
    test=dict(type=dataset_type, 
              data_root=data_root,
              pipeline=test_pipeline, 
              version=version, 
              test_mode=False),
    nonshuffler_sampler=dict(type='DistributedSampler'))
# optimizer
optimizer = dict(
    lr=0.002, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
total_epochs = 12
evaluation = dict(interval=2)


attack_severity_type = 'num_steps'
attack = dict(
    type='AutoPGD',
    epsilon=5.0,
    num_steps=[10],
    img_norm=img_norm_cfg,
    single_camera=False,
    mono_model=True,
    # loss_fn=dict(type='TargetedClassificationObjective', num_cls=len(class_names), random=True, thresh=0.1),
    # loss_fn=dict(type='LocalizationObjective',l2loss=False,loc=True,vel=True,orie=True),
    loss_fn=dict(type='ClassficationObjective', activate=False),
    category='Madry',
    rand_init=True,
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))


# attack_severity_type = 'initial_const'
# attack = dict(
#     type='CWAttack',
#     max_iterations=50,
#     learning_rate=25,
#     initial_const=[50],
#     img_norm=img_norm_cfg,
#     single_camera=False,
#     mono_model=True,
#     # loss_fn=dict(type='TargetedClassificationObjective', num_cls=len(class_names), random=True, thresh=0.1, targets=0),
#     loss_fn=dict(type='LocalizationObjective',l2loss=False,loc=True,vel=True,orie=True),
#     # loss_fn=dict(type='ClassficationObjective', activate=False),
#     assigner=dict(type='NuScenesAssigner', dis_thresh=4))

# attack_severity_type = 'epsilon'
# attack = dict(
#     type='FGSM',
#     epsilon=[5],
#     img_norm=img_norm_cfg,
#     single_camera=False,
#     mono_model=True,
#     # loss_fn=dict(type='TargetedClassificationObjective', num_cls=len(class_names), random=True, thresh=0.1, targets=0),
#     # loss_fn=dict(type='LocalizationObjective',l2loss=False,loc=True,vel=True,orie=True),
#     loss_fn=dict(type='ClassficationObjective', activate=False),
#     assigner=dict(type='NuScenesAssigner', dis_thresh=4))


# attack_severity_type = 'num_steps'
# attack = dict(
#     type='PGD',
#     epsilon=5,
#     step_size=0.1,
#     num_steps=[10],
#     img_norm=img_norm_cfg,
#     single_camera=False,
#     mono_model=True,
#     # loss_fn=dict(type='TargetedClassificationObjective', num_cls=len(class_names), random=True, thresh=0.1),
#     # loss_fn=dict(type='LocalizationObjective',l2loss=False,loc=True,vel=True,orie=True),
#     loss_fn=dict(type='ClassficationObjective', activate=False),
#     category='Madry',
#     rand_init=True,
#     assigner=dict(type='NuScenesAssigner', dis_thresh=4))

# attack_severity_type = 'scale'
# attack = dict(
#     type='PatchAttack',
#     step_size=5,
#     dynamic_patch_size=True,
#     scale=[0.1, 0.2, 0.3, 0.4],
#     num_steps=50,
#     mono_model=True,
#     # patch_size=(15,15),
#     img_norm=img_norm_cfg,
#     loss_fn=dict(type='LocalizationObjective',l2loss=False,loc=True,vel=True,orie=True),
#     assigner=dict(type='NuScenesAssigner', dis_thresh=4))

# class_names = [
#     'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
#     'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
# ]
# input_modality = dict(
#     use_lidar=False,
#     use_camera=True,
#     use_radar=False,
#     use_map=False,
#     use_external=False)
# attack_severity_type = 'scale'
# attack = dict(
#     type='UniversalPatchAttackOptim',
#     epoch=1,
#     lr=10,
#     is_train=False,
#     category_specify=False,
#     mono_model=True,
#     dataset_cfg=dict(
#         dataset=dict(type=dataset_type,
#                     data_root='../nuscenes_mini/',
#                     ann_file='../nuscenes_mini/' + 'nuscenes_infos_temporal_train_mono3d.coco.json',
#                     img_prefix='../nuscenes/',
#                     classes=class_names,
#                     pipeline=test_pipeline,
#                     modality=input_modality,
#                     test_mode=False,
#                     box_type_3d='Camera'),
#         shuffle=True,
#         workers_per_gpu=32),
#     dynamic_patch_size=True,
#     max_train_samples=323*6,
#     scale=[0.3],
#     patch_path=['/home/cixie/shaoyuan/BEV-Attack/zoo/BEVDet/uni_patch_new/BEVDet_Adv_coslr_size100_scale0.3_lr0.0392156862745098_sample323.pkl'],
#     patch_size=(100,100),
#     img_norm=img_norm_cfg,
#     loss_fn=dict(type='TargetedClassificationObjective',num_cls=10, random=True, thresh=0.1, targets=None),
#     assigner=dict(type='NuScenesAssigner', dis_thresh=4))