_base_ = [
    '../../_base_/models/bmn_400x100.py', '../../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'ActivityNetDataset'

data_root = 'data/apas_activity_net/videos'
data_root_val = data_root
ann_file_train = 'data/apas_activity_net/activitynet/splits/train.split3.bundle.json'
ann_file_val = 'data/apas_activity_net/activitynet/splits/test.split3.bundle.json'
ann_file_test = 'data/apas_activity_net/activitynet/splits/val.split3.bundle.json'

test_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(
        type='Collect',
        keys=['raw_feature'],
        meta_name='video_meta',
        meta_keys=[
            'video_name', 'duration_second', 'duration_frame', 'annotations',
            'feature_frame'
        ]),
    dict(type='ToTensor', keys=['raw_feature']),
]
train_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(type='GenerateLocalizationLabels'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=['video_name']),
    dict(type='ToTensor', keys=['raw_feature', 'gt_bbox']),
    dict(
        type='ToDataContainer',
        fields=[dict(key='gt_bbox', stack=False, cpu_only=True)])
]
val_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(type='GenerateLocalizationLabels'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=[
            'video_name', 'duration_second', 'duration_frame', 'annotations',
            'feature_frame'
        ]),
    dict(type='ToTensor', keys=['raw_feature', 'gt_bbox']),
    dict(
        type='ToDataContainer',
        fields=[dict(key='gt_bbox', stack=False, cpu_only=True)])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=8,
    train_dataloader=dict(drop_last=True),
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        data_prefix=data_root_val),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_prefix=data_root_val),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_root))
evaluation = dict(interval=1, metrics=['AR@AN'])

# optimizer
optimizer = dict(
    type='Adam', lr=0.001, weight_decay=0.0001)  # this lr is used for 2 gpus
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=7)
total_epochs = 9

# runtime settings
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
work_dir = './work_dirs/bmn_400x100_2x8_9e_activitynet_feature/'
output_config = dict(out=f'{work_dir}/results.json', output_format='json')
load_from = "https://download.openmmlab.com/mmaction/localization/bmn/bmn_400x100_9e_activitynet_feature/bmn_400x100_9e_activitynet_feature_20200619-42a3b111.pth"