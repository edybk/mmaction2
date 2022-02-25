model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3d',
        pretrained2d=True,
        pretrained='torchvision://resnet50',
        depth=50,
        conv1_kernel=(5, 7, 7),
        conv1_stride_t=2,
        pool1_stride_t=2,
        conv_cfg=dict(type='Conv3d'),
        norm_eval=False,
        inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        zero_init_residual=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=400,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01),
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[40, 80])
total_epochs = 200
checkpoint_config = dict(interval=5)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_256p_32x2x1_100e_kinetics400_rgb/i3d_r50_256p_32x2x1_100e_kinetics400_rgb_20200801-7d9f44de.pth"
resume_from = None
workflow = [('train', 1), ('val', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
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
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
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
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[144.7125, 132.8805, 124.7715],
        std=[65.127, 69.1305, 70.737],
        to_bgr=True),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(
        type='Normalize',
        mean=[144.7125, 132.8805, 124.7715],
        std=[65.127, 69.1305, 70.737],
        to_bgr=True),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
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
                clip_len=32,
                frame_interval=2,
                num_clips=1),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(
                type='MultiScaleCrop',
                input_size=224,
                scales=(1, 0.8),
                random_crop=False,
                max_wh_scale_gap=0),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
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
                clip_len=32,
                frame_interval=2,
                num_clips=1,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[144.7125, 132.8805, 124.7715],
                std=[65.127, 69.1305, 70.737],
                to_bgr=True),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]),
    test=dict(
        type='VideoDataset',
        ann_file='data/apas/splits/val.split1.txt',
        data_prefix='data/apas',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=2,
                num_clips=10,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='ThreeCrop', crop_size=256),
            dict(
                type='Normalize',
                mean=[144.7125, 132.8805, 124.7715],
                std=[65.127, 69.1305, 70.737],
                to_bgr=True),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
work_dir = './work_dirs/i3d_r50_32x2x1_100e_kinetics400_rgb_200epochs/'
gpu_ids = range(0, 1)
omnisource = False
module_hooks = []
