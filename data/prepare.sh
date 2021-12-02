##data split
#srun -p aipe --gres=gpu:0 --job-name='DGNet_data' -w SH-IDC2-172-20-21-216 \
#nohup python data_split.py \
#    --data_set='sidd' &
#
#srun -p aipe --gres=gpu:0 --job-name='DGNet_data' -w SH-IDC2-172-20-21-216 \
#nohup python data_split.py \
#    --data_set='renoir' &
#
#srun -p aipe --gres=gpu:0 --job-name='DGNet_data' -w SH-IDC2-172-20-21-216 \
#nohup python data_split.py \
#    --data_set='polyu' &


##prepare train data
#srun -p aipe --gres=gpu:0 --job-name='DGNet_data' -w SH-IDC2-172-20-21-72 \
#nohup python prepare_train_data.py \
#   --data_set='sidd' &

#srun -p aipe --gres=gpu:0 --job-name='DGNet_data' -w SH-IDC2-172-20-21-72 \
#nohup python prepare_train_data.py \
#   --data_set='renoir' &

#srun -p aipe --gres=gpu:0 --job-name='DGNet_data' -w SH-IDC2-172-20-21-72 \
#nohup python prepare_train_data.py \
#   --data_set='polyu' &

#srun -p aipe --gres=gpu:0 --job-name='DGNet_data' -w SH-IDC2-172-20-21-72 \
#nohup python prepare_test_data.py \
#   --data_set='sidd' &

#srun -p aipe --gres=gpu:0 --job-name='DGNet_data' -w SH-IDC2-172-20-21-72 \
#nohup python prepare_test_data.py \
#   --data_set='renoir' &

srun -p aipe --gres=gpu:0 --job-name='DGNet_data' -w SH-IDC2-172-20-21-72 \
nohup python prepare_test_data.py \
   --data_set='polyu' &