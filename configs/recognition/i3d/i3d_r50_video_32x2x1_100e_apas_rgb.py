_base_ = ['./i3d_r50_32x2x1_100e_kinetics400_rgb.py']

# dataset settings
# dataset_type = 'VideoDataset'
# data_root = 'data/kinetics400/videos_train'
# data_root_val = 'data/kinetics400/videos_val'
# ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'
# ann_file_val = 'data/kinetics400/kinetics400_val_list_videos.txt'
# ann_file_test = 'data/kinetics400/kinetics400_val_list_videos.txt'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

dataset_type = 'VideoDataset'
data_root = 'data/apas/frontal/clips'
data_root_val = 'data/apas/frontal/clips'
ann_file_train = 'data/apas/splits/train.split2.txt'
ann_file_val = 'data/apas/splits/val.split2.txt'
ann_file_test = 'data/apas/splits/test.split2.txt'
# img_norm_cfg = dict(
#     mean=[144.7125, 132.8805, 124.7715], std=[65.1270, 69.1305, 70.737], to_bgr=True)
img_norm_cfg = dict(
    mean=[144.01279543590812, 131.91472349201618, 123.31423661654405], std=[64.7040393831716, 68.7886688576991, 70.7488325943194], to_rgb=True)


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
    dict(type='Normalize', **img_norm_cfg),
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
    dict(type='Normalize', **img_norm_cfg),
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
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
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
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
