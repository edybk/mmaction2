import json

c3d_log_jsons="/data/home/bedward/workspace/mmpose-project/mmaction2/work_dirs/c3d_sports1m_16x1x1_45e_apas_rgb_200epochs/20220225_021103.log.json"
i3d_log_jsons = "/data/home/bedward/workspace/mmpose-project/mmaction2/work_dirs/i3d_r50_32x2x1_100e_kinetics400_rgb_200epochs/20220225_021154.log.json"
tsn_log_jsons = "/data/home/bedward/workspace/mmpose-project/mmaction2/work_dirs/tsn_r50_video_1x1x8_100e_kinetics400_rgb_apas/20220225_173001.log.json"

tsn_log2_jsons = "/data/home/bedward/workspace/mmpose-project/mmaction2/work_dirs/tsn_r50_video_1x1x8_100e_kinetics400_rgb_apas_withoutg0/20220605_152450.log.json"

csn_log_jsons = "/data/home/bedward/workspace/mmpose-project/mmaction2/work_dirs/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_apas/20220609_002728.log.json"


i3d_frontal_jsons = "/data/home/bedward/workspace/mmpose-project/mmaction2/work_dirs/i3d_r50_32x2x1_100e_apas_rgb_closeup/20220711_233245.log.json"

i3d_closeup_jsons = ""
def get_best(jsonsss):
    acc = []
    with open(jsonsss) as f:
        for l in f.readlines():
            d = json.loads(l)
            if "mode" in d and d['mode']=='val' and d['epoch']%5 == 0:
                acc.append((d.get("top1_acc", 0), d.get("epoch", "")))
    print(max(acc))
    return max(acc)
    

# get_best("/data/home/bedward/workspace/mmpose-project/mmaction2/work_dirs/i3d_r50_32x2x1_100e_apas_rgb_frontal_split1/20220716_130848.log.json")
get_best("/data/home/bedward/workspace/mmpose-project/mmaction2/work_dirs/c3d_sports1m_16x1x1_45e_apas_rgb_frontal_split1/20220716_130338.log.json")