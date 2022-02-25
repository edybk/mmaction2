model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='C3D',
        pretrained=
        'https://download.openmmlab.com/mmaction/recognition/c3d/c3d_sports1m_pretrain_20201016-dcc47ddc.pth',
        style='pytorch',
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),
        dropout_ratio=0.5,
        init_std=0.005),
    cls_head=dict(
        type='I3DHead',
        num_classes=101,
        in_channels=4096,
        spatial_type=None,
        dropout_ratio=0.5,
        init_std=0.01),
    train_cfg=None,
    test_cfg=dict(average_clips='score'))
dataset_type = 'VideoDataset'
data_root = 'data/apas'
data_root_val = 'data/apas'
ann_file_train = 'data/apas/splits/train.split1.txt'
ann_file_val = 'data/apas/splits/val.split1.txt'
ann_file_test = 'data/apas/splits/test.split1.txt'
img_norm_cfg = dict(
    mean=[144.7125, 132.8805, 124.7715],
    std=[65.127, 69.1305, 70.737],
    to_bgr=True)
split = 1
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=1, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(128, 171)),
    dict(type='RandomCrop', size=112),
    dict(type='Flip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[144.7125, 132.8805, 124.7715],
        std=[65.127, 69.1305, 70.737],
        to_bgr=True),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(128, 171)),
    dict(type='CenterCrop', crop_size=112),
    dict(
        type='Normalize',
        mean=[144.7125, 132.8805, 124.7715],
        std=[65.127, 69.1305, 70.737],
        to_bgr=True),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(128, 171)),
    dict(type='CenterCrop', crop_size=112),
    dict(
        type='Normalize',
        mean=[144.7125, 132.8805, 124.7715],
        std=[65.127, 69.1305, 70.737],
        to_bgr=True),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(
    videos_per_gpu=30,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='VideoDataset',
        ann_file='data/apas/splits/train.split1.txt',
        data_prefix='data/apas',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=16,
                frame_interval=1,
                num_clips=1),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(128, 171)),
            dict(type='RandomCrop', size=112),
            dict(type='Flip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[144.7125, 132.8805, 124.7715],
                std=[65.127, 69.1305, 70.737],
                to_bgr=True),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    val=dict(
        type='VideoDataset',
        ann_file='data/apas/splits/val.split1.txt',
        data_prefix='data/apas',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=16,
                frame_interval=1,
                num_clips=1,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(128, 171)),
            dict(type='CenterCrop', crop_size=112),
            dict(
                type='Normalize',
                mean=[144.7125, 132.8805, 124.7715],
                std=[65.127, 69.1305, 70.737],
                to_bgr=True),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    test=dict(
        type='VideoDataset',
        ann_file='data/apas/splits/test.split1.txt',
        data_prefix='data/apas',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=16,
                frame_interval=1,
                num_clips=10,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(128, 171)),
            dict(type='CenterCrop', crop_size=112),
            dict(
                type='Normalize',
                mean=[144.7125, 132.8805, 124.7715],
                std=[65.127, 69.1305, 70.737],
                to_bgr=True),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]))
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 200
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "https://download.openmmlab.com/mmaction/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth"
# resume_from = '/data/home/bedward/workspace/mmpose-project/mmaction2/work_dirs/c3d_sports1m_16x1x1_45e_apas_rgb/epoch_45.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]
work_dir = './work_dirs/c3d_sports1m_16x1x1_45e_apas_rgb_200epochs'
gpu_ids = range(0, 1)
omnisource = False
module_hooks = []
