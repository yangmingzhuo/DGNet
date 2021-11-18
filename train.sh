srun -p aipe --gres=gpu:2 --job-name='DGNet_data' -w SH-IDC2-172-20-21-216 \
nohup python train.py --gpus=2&