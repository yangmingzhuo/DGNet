srun -p aipe --gres=gpu:3 --job-name='DGNet_data' -w SH-IDC2-172-20-21-216 \
nohup python train.py \
    --gpus=3 \
    --data_set='renoir' \
    --ex_id=2 &