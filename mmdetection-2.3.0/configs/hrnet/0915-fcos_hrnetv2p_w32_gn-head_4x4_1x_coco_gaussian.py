_base_ = './fcos_hrnetv2p_w32_gn-head_4x4_1x_coco.py'
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w32',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(32, 64)),
            stage3=dict(num_channels=(32, 64, 128)),
            stage4=dict(num_channels=(32, 64, 128, 256)))),
    neck=dict(
        type='HRFPN',
        in_channels=[32, 64, 128, 256],
        out_channels=256,
        stride=1,
        num_outs=1),
    bbox_head=dict(
        type='FCOSHeadGaussian',
        num_classes=11,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        centerness_on_reg=False,
        strides=[4],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=5000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=1000)
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1088, 608), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1088, 608),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# dataset settings
dataset_type = 'DroneDataset'
data_root = '/home/sdb/wangshentao/myspace/thesis/data/visdrone2019/'
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'VisDrone2019-DET-train/instances_train2019.json',
        img_prefix=data_root + 'VisDrone2019-DET-train/train2019',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VisDrone2019-DET-val/instances_val2019.json',
        img_prefix=data_root + 'VisDrone2019-DET-val/val2019',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VisDrone2019-DET-val/instances_val2019.json',
        img_prefix=data_root + 'VisDrone2019-DET-val/val2019',
        pipeline=test_pipeline))
# data = dict(
#     samples_per_gpu=4,
#     workers_per_gpu=4,
#     train=dict(pipeline=train_pipeline),
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=8000,
    warmup_ratio=1.0 / 1000,
    step=[16, 24, 32])
total_epochs = 40
