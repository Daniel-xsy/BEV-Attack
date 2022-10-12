_base_ = [
    '../_base_/datasets/nus-mono3d.py', '../_base_/models/pgd.py',
    '../_base_/schedules/mmdet_schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='CustomFCOSMono3D',
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    bbox_head=dict(
        pred_bbox2d=False,
        group_reg_dims=(2, 1, 3, 1, 2,
                        4),  # offset, depth, size, rot, velo, bbox2d
        reg_branch=(
            (256, ),  # offset
            (256, ),  # depth
            (256, ),  # size
            (256, ),  # rot
            (),  # velo
            (256, )  # bbox2d
        ),
        loss_depth=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        bbox_coder=dict(
            type='PGDBBoxCoder',
            base_depths=((31.99, 21.12), (37.15, 24.63), (39.69, 23.97),
                         (40.91, 26.34), (34.16, 20.11), (22.35, 13.70),
                         (24.28, 16.05), (27.26, 15.50), (20.61, 13.68),
                         (22.74, 15.01)),
            base_dims=((4.62, 1.73, 1.96), (6.93, 2.83, 2.51),
                       (12.56, 3.89, 2.94), (11.22, 3.50, 2.95),
                       (6.68, 3.21, 2.85), (6.68, 3.21, 2.85),
                       (2.11, 1.46, 0.78), (0.73, 1.77, 0.67),
                       (0.41, 1.08, 0.41), (0.50, 0.99, 2.52)),
            code_size=9)),
    # set weight 1.0 for base 7 dims (offset, depth, size, rot)
    # 0.05 for 2-dim velocity and 0.2 for 4-dim 2D distance targets
    train_cfg=dict(code_weight=[
        1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2
    ]),
    test_cfg=dict(nms_pre=1000, nms_thr=0.8, score_thr=0.01, max_per_img=200))

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
            dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img']),
        ])
]
version = 'v1.0-mini'
dataset_type = 'CustomNuScenesMonoDataset'
data_root = '/data1/data/shaoyuan/nuscenes_mini/'
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
    lr=0.004, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
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
evaluation = dict(interval=4)
runner = dict(max_epochs=total_epochs)


# attack = dict(
#     type='UniversalPatchAttack',
#     step_size=5,
#     epoch=10,
#     loader=dict(type=dataset_type,
#              data_root=data_root,
#              ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
#              pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
#              test_mode=False,
#              adv_mode=True,
#              classes=class_names, modality=input_modality),
#     loss_fn=dict(type='ClassficationObjective', activate=False),
#     assigner=dict(type='NuScenesAssigner', dis_thresh=4),
#     category_specify=True,
#     catagory_num=10,
#     patch_size=(15,15),
#     dynamic_patch_size=False,
#     scale=0.5,
#     img_norm=img_norm_cfg,
# )

# attack = dict(
#     type='PatchAttack',
#     step_size=[5/57.375, 5/57.120, 5/58.395],
#     dynamic_patch_size=False,
#     scale=0.4,
#     num_steps=50,
#     patch_size=(15,15),
#     img_norm=img_norm_cfg,
#     loss_fn=dict(type='ClassficationObjective', activate=False),
#     assigner=dict(type='NuScenesAssigner', dis_thresh=4))

attack = dict(
    type='PGD',
    epsilon=5,
    step_size=0.1,
    num_steps=8,
    img_norm=img_norm_cfg,
    single_camera=False,
    mono_model=True,
    loss_fn=dict(type='ClassficationObjective', activate=False),
    category='Madry',
    rand_init=True,
    assigner=dict(type='NuScenesAssigner', dis_thresh=4))