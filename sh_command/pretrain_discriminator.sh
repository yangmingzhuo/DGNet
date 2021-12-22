srun -p aipe --gres=gpu:2 --job-name='DGNet' -w SH-IDC2-172-20-21-216 \
nohup python train_singleDataset.py \
    --gpus=0,1 \
    --data_set1='renoir_v2' \
    --data_set1='nind' \
    --data_set1='rid2021_v2' \
    --batch_size=512 \
    --lr=1e-2 \
    --lr_min=1e-4 \
    --exp_id=0 &