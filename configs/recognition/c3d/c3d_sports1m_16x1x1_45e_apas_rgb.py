_base_ = '../../_base_/models/c3d_sports1m_pretrained.py'

# dataset settings
# dataset_type = 'ActivityNetDataset'
# data_root = '/data/home/bedward/datasets/APAS-Activities-Eddie/videos/frontal_view'
# data_root_val = '/data/home/bedward/datasets/APAS-Activities-Eddie/videos/frontal_view'

# ann_file_train = '/data/home/bedward/datasets/APAS-Activities-Eddie/annotations/gestures_activitynet/train_annotations.json'
# ann_file_val = '/data/home/bedward/datasets/APAS-Activities-Eddie/annotations/gestures_activitynet/val_annotations.json'
# ann_file_test = '/data/home/bedward/datasets/APAS-Activities-Eddie/annotations/gestures_activitynet/test_annotations.json'
# img_norm_cfg = dict(
#     mean=[144.7125, 132.8805, 124.7715], std=[65.1270, 69.1305, 70.737], to_bgr=True)

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/apas'
data_root_val = 'data/apas'
ann_file_train = 'data/apas/splits/train.split1.txt'
ann_file_val = 'data/apas/splits/val.split1.txt'
ann_file_test = 'data/apas/splits/test.split1.txt'
img_norm_cfg = dict(
    mean=[144.7125, 132.8805, 124.7715], std=[65.1270, 69.1305, 70.737], to_bgr=True)

split = 1  # official train/test splits. valid numbers: 1, 2, 3

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=1, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(128, 171)),
    dict(type='RandomCrop', size=112),
    
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
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
    dict(type='Normalize', **img_norm_cfg),
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
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(
    videos_per_gpu=30,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9,
    weight_decay=0.0005)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 45
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 10), ('val', 1)]
