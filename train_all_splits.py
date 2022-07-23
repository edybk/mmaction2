import subprocess
import os

#i3d
frontal_i3d_config_prefix = "/data/home/bedward/workspace/mmpose-project/mmaction2/configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb_apas_frontal_split"
closeup_i3d_config_prefix = "/data/home/bedward/workspace/mmpose-project/mmaction2/configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb_apas_closeup_split"

#c3d
frontal_c3d_config_prefix = "/data/home/bedward/workspace/mmpose-project/mmaction2/configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb_frontal_split"
closeup_c3d_config_prefix = "/data/home/bedward/workspace/mmpose-project/mmaction2/configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb_closeup_split"

# config_prefix = frontal_i3d_config_prefix
config_prefix = closeup_i3d_config_prefix
# config_prefix = frontal_c3d_config_prefix
for split in range(1, 6):
    out = subprocess.check_output(f"python tools/train.py {config_prefix}{split}.py", stderr=subprocess.STDOUT, shell=True)
    print(out)