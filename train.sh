#sidd
#srun -p aipe --gres=gpu:2 --job-name='DGNet' -w SH-IDC2-172-20-21-72 \
#nohup python train_singleDataset.py \
#    --gpus=0,1 \
#    --data_set='polyu' \
#    --exp_id=0 &

#renoir
#srun -p aipe --gres=gpu:2 --job-name='DGNet' -w SH-IDC2-172-20-21-216 \
#nohup python train_singleDataset.py \
#    --gpus=2,3 \
#    --data_set='renoir' \
#    --exp_id=0 &

#polyu
srun -p aipe --gres=gpu:2 --job-name='DGNet' -w SH-IDC2-172-20-21-216 \
nohup python train_singleDataset.py \
    --gpus=4,5 \
    --data_set='polyu' \
    --exp_id=0 &