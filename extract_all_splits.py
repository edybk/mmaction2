import argparse
import subprocess
import json
import os
import numpy as np
from video_feature_extraction import main as export_videos

rawframes_frontal = "data/apas/rawframes_raw"
rawframes_closeup = "data/apas/rawframes_raw_closeup"

def get_best(jsonsss):
    acc = []
    with open(jsonsss) as f:
        for l in f.readlines():
            d = json.loads(l)
            if "mode" in d and d['mode']=='val' and d['epoch']%5 == 0:
                acc.append((d.get("top1_acc", 0), d.get("epoch", "")))
    print(max(acc))
    return max(acc)

def get_best_checkpoint(work_dir):
    acc = []
    for j in os.listdir(work_dir):
        if not j.endswith(".json"):
            continue
        try:
            acc.append(get_best(f"{work_dir}/{j}"))
        except:
            pass
    acc, epoch = max(acc)
    return f"{work_dir}/epoch_{epoch}.pth", acc

def extract(args, model_type = "tsn", view = "frontal"):
    if model_type == "c3d":
        model_name = f"new_c3d_sports1m_16x1x1_45e_apas_rgb_{view}"
    elif model_type == "tsn":
        model_name = "tsn_r50_video_1x1x8_100e_kinetics400_rgb_apas_withoutg0"
    elif model_type=="csn":
        model_name = "ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_apas"
    elif model_type == "i3d":
        if view == "frontal":
            model_name= "i3d_r50_32x2x1_100e_apas_rgb_frontal"
        elif view == "closeup":
            model_name= "i3d_r50_32x2x1_100e_apas_rgb_closeup"

    if view == "frontal":
        rawframes_root = rawframes_frontal
    elif view == "closeup":
        rawframes_root = rawframes_closeup
    else:
        raise ""
        
        
    accs = []
    args.modality = "RGB"
    args.data_prefix = rawframes_root
    for split in range(1, 6):
        work_dir = f"./work_dirs/{model_name}_split{split}"
        ckpt, acc = get_best_checkpoint(work_dir)
        accs.append(acc)
        print(ckpt)
        args.ckpt = ckpt
        args.model_choice = model_type
        for sset in ["train", "val", "test"]:
            args.data_prefix = rawframes_root
            args.data_list = f"data/apas_activity_net/splits/{sset}.split{split}.bundle"
            args.output_prefix = f"data/apas_activity_net/rgb_feat/{view}/{model_type}/split{split}"
            export_videos(args)
            # break
        # break
    print(f"mean validation: {np.mean(accs)}")
    
# extract(model_type = "tsn")
# extract(model_type = "c3d")
# extract(model_type = "csn")

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
    parser.add_argument('--model-choice', default='tsn', choices=['tsn', 'c3d', 'csn', 'i3d'])
    parser.add_argument('--view', default='frontal', choices=['frontal', 'closeup', 'csn', 'i3d'])
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

args = parse_args()


extract(args, model_type = "i3d", view="closeup")