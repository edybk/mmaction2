import subprocess
import json
import os
import numpy as np


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

def extract(model_type = "tsn"):
    if model_type == "c3d":
        model_name = "c3d_sports1m_16x1x1_45e_apas_rgb_200epochs_withoutg0"
    elif model_type == "tsn":
        model_name = "tsn_r50_video_1x1x8_100e_kinetics400_rgb_apas_withoutg0"
    elif model_type=="csn":
        model_name = "ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_apas"
    elif model_type == "i3d":
        model_name = "i3d_r50_32x2x1_100e_kinetics400_rgb_200epochs"
        raise 'NOT IMPLEMENTED'
        
        
    accs = []
    for split in range(1, 6):
        work_dir = f"./work_dirs/{model_name}_split{split}"
        ckpt, acc = get_best_checkpoint(work_dir)
        accs.append(acc)
        print(ckpt)
        for sset in ["train", "val", "test"]:
            
            cmd = f"python tools/data/activitynet/tsn_feature_extraction.py --data-prefix data/apas_activity_net/rawframes --data-list data/apas_activity_net/splits/{sset}.split{split}.bundle --output-prefix data/apas_activity_net/rgb_feat/{model_type}/split{split} --modality RGB --ckpt {ckpt} --model-choice {model_type}"
            # print(cmd)
            # out = subprocess.check_output(cmd, shell=True)
            # print(out)
            # break
        # break
    print(f"mean validation: {np.mean(accs)}")
    
extract(model_type = "tsn")
extract(model_type = "c3d")
extract(model_type = "csn")