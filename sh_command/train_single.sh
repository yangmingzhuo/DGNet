#train with single dataset
#sidd
#srun -p aipe --gres=gpu:2 --job-name='DGNet' -w SH-IDC2-172-20-21-72 \
#nohup python train_singleDataset.py \
#    --gpus=0,1 \
#    --data_set='sidd' \
#    --nEpochs=1 \
#    --exp_id=0 &

#renoir
#srun -p aipe --gres=gpu:2 --job-name='DGNet' -w SH-IDC2-172-20-21-72 \
#nohup python train_singleDataset.py \
#    --gpus=2,3 \
#    --data_set='renoir' \
#    --exp_id=0 &

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
srun -p aipe --gres=gpu:2 --job-name='DGNet' -w SH-IDC2-172-20-21-216 \
nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=23334 train_singleDataset_ddp.py \
    --gpus=0,1 \
    --data_set='sidd' \
    --batch_size=16 \
    --exp_id=0 &
