import subprocess
import os
import pickle
import numpy as np


view1_dir = "/data/home/bedward/datasets/APAS-Activities-Eddie/rgb_feat/closeup/i3d/"
view2_dir = "/data/home/bedward/datasets/APAS-Activities-Eddie/rgb_feat/frontal/i3d/"

final_features_exprt_dir = "/data/home/bedward/datasets/APAS-Activities-Eddie/rgb_feat/both/i3d/"

os.makedirs(final_features_exprt_dir, exist_ok=True)
for split in range(1, 6):
    split_export_dir = f"{final_features_exprt_dir}/split{split}"
    os.makedirs(split_export_dir, exist_ok=True)
    tsn_split_features1 = view1_dir + '/split' + str(split)
    tsn_split_features2 = view2_dir + '/split' + str(split)
    for fbasename1, fbasename2 in zip(os.listdir(tsn_split_features1), os.listdir(tsn_split_features2)):
        arr1 = np.load(f"{tsn_split_features1}/{fbasename1}")
        # with open(f"{tsn_split_features1}/{fbasename1}", 'rb') as f:
        #     arr1 = pickle.load(f)
        print(arr1.shape)

        arr2 = np.load(f"{tsn_split_features2}/{fbasename2}")

        # with open(f"{tsn_split_features2}/{fbasename2}", 'rb') as f:
        #     arr2 = pickle.load(f)
        print(arr2.shape)
        
        l = min(arr1.shape[1], arr2.shape[1])
        
        
        combined = np.concatenate((arr1[:, :l], arr2[:, :l]), axis=0)
        print(combined.shape)
        np.save(split_export_dir + '/' + fbasename1.split(".")[0], combined)
    