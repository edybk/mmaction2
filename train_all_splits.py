import subprocess

for split in range(1, 6):
    out = subprocess.check_output(f"python tools/train.py /data/home/bedward/workspace/mmpose-project/mmaction2/configs/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_apas_split{split}.py &> logs/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_apas_split{split}.txt", shell=True)
    print(out)