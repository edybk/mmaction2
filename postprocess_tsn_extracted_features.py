import subprocess
import os
import pickle
import numpy as np

repeat_size = 16
model_type = "tsn"
tsn_features_export_dir = "/data/home/bedward/workspace/mmpose-project/mmaction2/data/apas_activity_net/rgb_feat/tsn"
final_features_exprt_dir = "/data/home/bedward/datasets/APAS-Activities-Eddie/new_rgb_feat_6_6_2022" + '/' + model_type

os.makedirs(final_features_exprt_dir, exist_ok=True)
for split in range(1, 6):
    split_export_dir = f"{final_features_exprt_dir}/split{split}"
    os.makedirs(split_export_dir, exist_ok=True)
    tsn_split_features = tsn_features_export_dir + '/split' + str(split)
    for fbasename in os.listdir(tsn_split_features):
        with open(f"{tsn_split_features}/{fbasename}", 'rb') as f:
            arr = pickle.load(f)
        print(arr.shape)

        inflated_arr = np.repeat(arr, repeat_size, axis=0)
        inflated_arr = inflated_arr.transpose()
        print(inflated_arr.shape)
        # break
        np.save(split_export_dir + '/' + fbasename.split(".")[0], inflated_arr)