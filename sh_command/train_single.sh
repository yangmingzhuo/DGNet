#train with single dataset
#sidd
#srun -p aipe --gres=gpu:2 --job-name='DGNet' -w SH-IDC2-172-20-21-72 \
#nohup python train_singleDataset.py \
#    --gpus=0,1 \
#    --data_set='sidd' \
#    --exp_id=0 &

#renoir
srun -p aipe --gres=gpu:2 --job-name='DGNet' -w SH-IDC2-172-20-21-216 \
nohup python train_singleDataset.py \
    --gpus=0,1 \
    --data_set='renoir_v2' \
    --log_dir='logs_v2' \
    --pretrain_model='/mnt/lustre/yangmingzhuo/DGNet/logs_v2/model_ELU_UNet_gpu_2,3_ds_renoir_v2_ps_128_bs_32_ep_150_lr_0.0002_lr_min_1e-05_exp_id_0/checkpoint/model_latest.pth' \
    --exp_id=0 &

#polyu
#srun -p aipe --gres=gpu:2 --job-name='DGNet' -w SH-IDC2-172-20-21-216 \
#nohup python train_singleDataset.py \
#    --gpus=4,5 \
#    --data_set='polyu' \
#    --exp_id=0 &

#nind
#srun -p aipe --gres=gpu:2 --job-name='DGNet' -w SH-IDC2-172-20-21-216 \
#nohup python train_singleDataset.py \
#    --gpus=0,1 \
#    --data_set='nind' \
#    --exp_id=0 &

#rid2021
#srun -p aipe --gres=gpu:2 --job-name='DGNet' -w SH-IDC2-172-20-21-216 \
#nohup python train_singleDataset.py \
#    --gpus=0,1 \
#    --data_set='rid2021_v2' \
#    --exp_id=0 &

##DDP
#sidd
#srun -p aipe --gres=gpu:4 --job-name='DGNet' -w SH-IDC2-172-20-21-216 \
#nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=23334 train_singleDataset_ddp.py \
#    --gpus=4,5,6,7 \
#    --data_set='sidd' \
#    --batch_size=8 \
#    --exp_id=0 &
