srun -p aipe --gres=gpu:1 --job-name='DGNet' -w SH-IDC2-172-20-21-216 \
nohup python validation.py \
    --data_set='renoir_v2' \
    --pretrained='/mnt/lustre/yangmingzhuo/DGNet/logs_v2/model_ELU_UNet_gpu_4,5,6,7_ds_sidd_nind_rid2021_v2_td_renoir_v2_ps_128_bs_8_ep_150_lr_0.0002_lr_min_1e-05_lam_exp_id_1_ddp/checkpoint/model_best.pth' \
    --save_imgs=0 \
    --gpus=2 &
