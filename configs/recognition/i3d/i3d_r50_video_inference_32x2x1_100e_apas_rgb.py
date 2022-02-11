_base_ = [
    '../../_base_/models/i3d_r50.py', '../../_base_/schedules/sgd_100e.py',
    '../../_base_/default_runtime.py'
]
# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/apas'
data_root_val = 'data/apas'
ann_file_train = 'data/apas/train.split1.txt'
ann_file_val = 'data/apas/val.split1.txt'
ann_file_test = 'data/apas/test.split1.txt'
img_norm_cfg = dict(
    mean=[144.7125, 132.8805, 124.7715], std=[65.1270, 69.1305, 70.737], to_bgr=True)
test_pipeline = [
    dict(type='DecordInit', num_threads=1),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=1,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        ann_file=None,
        data_prefix=None,
        pipeline=test_pipeline))
