import subprocess
import json
import os

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
    return f"{work_dir}/epoch_{epoch}.pth"

def extract(model_type = "tsn"):
    if model_type == "c3d":
        model_name = "c3d_sports1m_16x1x1_45e_apas_rgb_200epochs_withoutg0"
    elif model_type == "tsn":
        model_name = "tsn_r50_video_1x1x8_100e_kinetics400_rgb_apas_withoutg0"
        
    for split in range(1, 6):
        work_dir = f"./work_dirs/{model_name}_split{split}"
        ckpt = get_best_checkpoint(work_dir)
        print(ckpt)
        for sset in ["train", "val", "test"]:
            
            cmd = f"python tools/data/activitynet/tsn_feature_extraction.py --data-prefix data/apas_activity_net/rawframes --data-list data/apas_activity_net/splits/{sset}.split{split}.bundle --output-prefix data/apas_activity_net/rgb_feat/{model_type}/split{split} --modality RGB --ckpt {ckpt} --model-choice {model_type}"
            
            out = subprocess.check_output(cmd, shell=True)
            print(out)
            # break
        # break
    
# extract(model_type = "tsn")
extract(model_type = "c3d")