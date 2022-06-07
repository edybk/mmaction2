# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import pickle

import mmcv
import numpy as np
import torch

from mmaction.datasets.pipelines import Compose
from mmaction.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Extract TSN Feature')
    parser.add_argument('--data-prefix', default='', help='dataset prefix')
    parser.add_argument('--output-prefix', default='', help='output prefix')
    parser.add_argument(
        '--data-list',
        help='video list of the dataset, the format should be '
        '`frame_dir num_frames output_file`')
    parser.add_argument(
        '--frame-interval',
        type=int,
        default=16,
        help='the sampling frequency of frame in the untrimed video')
    parser.add_argument('--modality', default='RGB', choices=['RGB', 'Flow'])
    parser.add_argument('--model-choice', default='tsn', choices=['tsn', 'c3d'])
    parser.add_argument('--ckpt', help='checkpoint for feature extraction')
    parser.add_argument(
        '--part',
        type=int,
        default=0,
        help='which part of dataset to forward(alldata[part::total])')
    parser.add_argument(
        '--total', type=int, default=1, help='how many parts exist')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.is_rgb = args.modality == 'RGB'
    if args.model_choice == "tsn":
        args.clip_len = 1 if args.is_rgb else 5
        args.input_format = 'NCHW' if args.is_rgb else 'NCHW_Flow'
        rgb_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_bgr=False)
        the_scale = (-1, 256)
        the_crop_size = 256
    elif args.model_choice == "c3d":
        args.input_format = "NCTHW"
        rgb_norm_cfg = dict(
            mean=[144.7125, 132.8805, 124.7715],
            std=[65.127, 69.1305, 70.737],
            to_bgr=True)
        the_scale = (128, 171)
        the_crop_size = 112
        args.clip_len = 16


    flow_norm_cfg = dict(mean=[128, 128], std=[128, 128])
    args.img_norm_cfg = rgb_norm_cfg if args.is_rgb else flow_norm_cfg
    args.f_tmpl = 'img_{:05d}.jpg' if args.is_rgb else 'flow_{}_{:05d}.jpg'
    args.in_channels = args.clip_len * (3 if args.is_rgb else 2)
    # max batch_size for one forward
    args.batch_size = 200

    # define the data pipeline for Untrimmed Videos
    data_pipeline = [
        dict(
            type='UntrimmedSampleFrames',
            clip_len=args.clip_len,
            frame_interval=args.frame_interval,
            start_index=0),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=the_scale),
        dict(type='CenterCrop', crop_size=the_crop_size),
        dict(type='Normalize', **args.img_norm_cfg),
        dict(type='FormatShape', input_format=args.input_format),
        dict(type='Collect', keys=['imgs'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    data_pipeline = Compose(data_pipeline)

    if args.model_choice == 'tsn':
        
        # define TSN R50 model, the model is used as the feature extractor
        model_cfg = dict(
            type='Recognizer2D',
            backbone=dict(
                type='ResNet',
                pretrained='torchvision://resnet50',
                depth=50,
                norm_eval=False),
            cls_head=dict(
                type='TSNHead',
                num_classes=6,
                in_channels=2048,
                spatial_type='avg',
                consensus=dict(type='AvgConsensus', dim=1),
                dropout_ratio=0.4,
                init_std=0.01),
            # model training and testing settings
            train_cfg=None,
            test_cfg=dict(average_clips=None, 
                          feature_extraction=True))
    elif args.model_choice == "c3d":
        model_cfg = dict(
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
                num_classes=6,
                in_channels=4096,
                spatial_type=None,
                dropout_ratio=0.5,
                init_std=0.01),
            train_cfg=None,
            test_cfg=dict(average_clips='score', feature_extraction=True))
    
    model = build_model(model_cfg)
    # load pretrained weight into the feature extractor
    state_dict = torch.load(args.ckpt)['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    data = open(args.data_list).readlines()
    data = [x.strip() for x in data]
    data = data[args.part::args.total]

    # enumerate Untrimmed videos, extract feature from each of them
    prog_bar = mmcv.ProgressBar(len(data))
    if not osp.exists(args.output_prefix):
        os.system(f'mkdir -p {args.output_prefix}')

    for item in data:
        frame_dir = item
        output_file = osp.basename(frame_dir) + '.pkl'
        frame_dir = osp.join(args.data_prefix, frame_dir)
        output_file = osp.join(args.output_prefix, output_file)
        length = len([ftmp for ftmp in os.listdir(frame_dir) if "img" in ftmp])
        assert output_file.endswith('.pkl')
        length = int(length)

        # prepare a pseudo sample
        tmpl = dict(
            frame_dir=frame_dir,
            total_frames=length,
            filename_tmpl=args.f_tmpl,
            start_index=0,
            modality=args.modality)
        sample = data_pipeline(tmpl)
        imgs = sample['imgs']
        shape = imgs.shape
        # the original shape should be N_seg * C * H * W, resize it to N_seg *
        # 1 * C * H * W so that the network return feature of each frame (No
        # score average among segments)
        imgs = imgs.reshape((shape[0], 1) + shape[1:])
        imgs = imgs.cuda()

        def forward_data(model, data):
            # chop large data into pieces and extract feature from them
            results = []
            start_idx = 0
            num_clip = data.shape[0]
            while start_idx < num_clip:
                with torch.no_grad():
                    part = data[start_idx:start_idx + args.batch_size]
                    feat = model.forward(part, return_loss=False)
                    results.append(feat)
                    start_idx += args.batch_size
            return np.concatenate(results)

        feat = forward_data(model, imgs)
        with open(output_file, 'wb') as fout:
            pickle.dump(feat, fout)
        prog_bar.update()


if __name__ == '__main__':
    main()
