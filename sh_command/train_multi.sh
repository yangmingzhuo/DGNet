#train with multi dataset
#sidd
#srun -p aipe --gres=gpu:2 --job-name='DGNet' -w SH-IDC2-172-20-21-72 \
#nohup python train_multiDataset.py \
#    --gpus=4,5 \
#    --data_set1='renoir' \
#    --data_set2='nind' \
#    --data_set3='rid2021_v2' \
#    --data_set_test='sidd' \
#    --exp_id=0 &

#renoir
#srun -p aipe --gres=gpu:2 --job-name='DGNet' -w SH-IDC2-172-20-21-72 \
#nohup python train_multiDataset.py \
#    --gpus=6,7 \
#    --data_set1='sidd' \
#    --data_set2='nind' \
#    --data_set3='rid2021_v2' \
#    --data_set_test='renoir' \
#    --exp_id=0 &

#nind
#srun -p aipe --gres=gpu:2 --job-name='DGNet' -w SH-IDC2-172-20-21-216 \
#nohup python train_multiDataset.py \
#    --gpus=4,5 \
#    --data_set1='sidd' \
#    --data_set2='renoir' \
#    --data_set3='rid2021_v2' \
#    --data_set_test='nind' \
#    --exp_id=0 &

#rid2021
#srun -p aipe --gres=gpu:2 --job-name='DGNet' -w SH-IDC2-172-20-21-216 \
#nohup python train_multiDataset.py \
#    --gpus=6,7 \
#    --data_set1='sidd' \
#    --data_set2='renoir' \
#    --data_set3='nind' \
#    --data_set_test='rid2021_v2' \
#    --exp_id=0 &


#DDP
#sidd
#srun -p aipe --gres=gpu:4 --job-name='DGNet' -w SH-IDC2-172-20-21-72 \
#nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=10315 train_multiDataset_ddp_ours.py \
#    --gpus=4,5,6,7 \
#    --data_set1='renoir' \
#    --data_set2='nind' \
#    --data_set3='rid2021_v2' \
#    --data_set_test='sidd' \
#    --batch_size=8 \
#    --lambda_ad=0 \
#    --exp_id=1 &

srun -p aipe --gres=gpu:4 --job-name='DGNet' -w SH-IDC2-172-20-21-72 \
nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=10316 train_multiDataset_ddp.py \
    --gpus=0,1,2,3 \
    --data_set1='sidd' \
    --data_set2='renoir' \
    --data_set3='rid2021_v2' \
    --data_set_test='nind' \
    --batch_size=8 \
    --lambda_ad=0 \
    --exp_id=1 &
