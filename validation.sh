srun -p aipe --gres=gpu:2 --job-name='DGNet' -w SH-IDC2-172-20-21-72 \
nohup python validation.py \
    --data_set='rid2021_v2' \
    --pretrained='/mnt/lustre/yangmingzhuo/DGNet/logs_v2/model_ELU_UNet_gpu_0,1_ds_nind_ps_128_bs_32_ep_150_lr_0.0002_lr_min_1e-05_exp_id_0/checkpoint' \
    --gpus=6 &
