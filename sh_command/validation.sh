srun -p aipe --gres=gpu:1 --job-name='DGNet' -w SH-IDC2-172-20-21-216 \
nohup python validation.py \
    --data_set='renoir_v2' \
    --pretrained='/mnt/lustre/yangmingzhuo/DGNet/logs_v2/baseline/model_ELU_UNet_gpu_6,7_ds_sidd_nind_rid2021_v2_td_renoir_ps_128_bs_32_ep_150_lr_0.0002_lr_min_1e-05_exp_id_0/checkpoint/model_best.pth' \
    --save_imgs=0 \
    --gpus=0 &
