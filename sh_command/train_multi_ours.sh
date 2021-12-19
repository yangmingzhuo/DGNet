# train with multi dataset
#DDP
#sidd
srun -p aipe --gres=gpu:4 --job-name='DGNet' -w SH-IDC2-172-20-21-216 \
nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=10314 train_multiDataset_ddp_ours.py \
    --gpus=4,5,6,7 \
    --data_set1='renoir_v2' \
    --data_set2='nind' \
    --data_set3='rid2021_v2' \
    --data_set_test='sidd' \
    --batch_size=8 \
    --lr_ad=2e-4 \
    --lr_min_ad=1e-5 \
    --lambda_ad=0.5 \
    --lambda_kl=0.0 \
    --temperature=20.0 \
    --exp_id=2 &
