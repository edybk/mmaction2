import json
import glob
import shutil
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
    return f"{work_dir}/epoch_{epoch}.pth", acc
    

work_dir = "/data/home/bedward/workspace/mmpose-project/mmaction2/work_dirs/c3d_sports1m_16x1x1_45e_apas_rgb_frontal_split1"
work_dir = "/data/home/bedward/workspace/mmpose-project/mmaction2/work_dirs/c3d_sports1m_16x1x1_45e_apas_rgb_frontal_split2"
work_dir = "/data/home/bedward/workspace/mmpose-project/mmaction2/work_dirs/c3d_sports1m_16x1x1_45e_apas_rgb_frontal_split3"

for wd in os.listdir('/data/home/bedward/workspace/mmpose-project/mmaction2/work_dirs/'):
    work_dir = '/data/home/bedward/workspace/mmpose-project/mmaction2/work_dirs/' + wd
    # jsons = list(glob.glob(f"{work_dir}/*.json"))[-1]

    try:
        
        best = get_best_checkpoint(work_dir)
        os.makedirs(f"{work_dir}/saved/", exist_ok=True)
        shutil.copyfile(best[0], f"{work_dir}/saved/{os.path.basename(best[0])}.best")
        for f in glob.glob(f"{work_dir}/*.pth"):
            os.remove(f)
    except Exception as e:
        print(e)