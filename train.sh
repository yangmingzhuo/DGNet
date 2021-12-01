#sidd
srun -p aipe --gres=gpu:2 --job-name='DGNet' -w SH-IDC2-172-20-21-216 \
nohup python train_singleDataset.py \
    --gpus=1,2 \
    --data_set='sidd' \
    --nEpochs=200 \
    --batch_size=32 &

#renoir
#srun -p aipe --gres=gpu:1 --job-name='DGNet' -w SH-IDC2-172-20-21-216 \
#nohup python train_singleDataset.py \
#    --gpus=1 \
#    --data_set='renoir' \
#    --random &