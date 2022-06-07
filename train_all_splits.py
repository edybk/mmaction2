import subprocess

for split in range(1, 6):
    out = subprocess.check_output(f"python tools/train.py /data/home/bedward/workspace/mmpose-project/mmaction2/configs/recognition/tsn/tsn_r50_video_1x1x8_100e_apas_rgb_withoutg0_split{split}.py &> logs/tsn_r50_video_1x1x8_100e_apas_rgb_200epochs_withoutg0_split{split}.txt", shell=True)
    print(out)