import json
acc = []
c3d_log_jsons="/data/home/bedward/workspace/mmpose-project/mmaction2/work_dirs/c3d_sports1m_16x1x1_45e_apas_rgb_200epochs/20220225_021103.log.json"
i3d_log_jsons = "/data/home/bedward/workspace/mmpose-project/mmaction2/work_dirs/i3d_r50_32x2x1_100e_kinetics400_rgb_200epochs/20220225_021154.log.json"
with open(c3d_log_jsons) as f:
    for l in f.readlines():
        d = json.loads(l)
        if "mode" in d and d['mode']=='val' and d['epoch']%5 == 0:
            acc.append((d.get("top1_acc", 0), d.get("epoch", "")))
print(max(acc))