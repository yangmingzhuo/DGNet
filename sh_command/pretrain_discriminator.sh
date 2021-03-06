srun -p aipe --gres=gpu:2 --job-name='DGNet' -w SH-IDC2-172-20-21-216 \
nohup python train_multiDataset_discriminator.py \
    --gpus=0 \
    --data_set1='renoir_v2' \
    --data_set2='nind' \
    --data_set3='rid2021_v2' \
    --batch_size=256 \
    --lr=2e-4 \
    --lr_min=1e-5 \
    --exp_id=0 &