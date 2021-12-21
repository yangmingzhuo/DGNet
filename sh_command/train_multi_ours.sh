# train with multi dataset
#DDP
#sidd
#srun -p aipe --gres=gpu:4 --job-name='DGNet' -w SH-IDC2-172-20-21-216 \
#nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=10313 train_multiDataset_ddp_ours_v2.py \
#    --gpus=4,5,6,7 \
#    --data_set1='renoir_v2' \
#    --data_set2='nind' \
#    --data_set3='rid2021_v2' \
#    --data_set_test='sidd' \
#    --batch_size=8 \
#    --lr_ad=1e-2 \
#    --lr_min_ad=1e-3 \
#    --lambda_ad=0.01 \
#    --lambda_kl=0.0 \
#    --temperature=20.0 \
#    --exp_id='v3' &

srun -p aipe --gres=gpu:4 --job-name='DGNet' -w SH-IDC2-172-20-21-72 \
nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=10312 train_multiDataset_ddp_ours.py \
    --gpus=4,5,6,7 \
    --data_set1='renoir_v2' \
    --data_set2='nind' \
    --data_set3='rid2021_v2' \
    --data_set_test='sidd' \
    --pretrain_model='/mnt/lustre/yangmingzhuo/DGNet/logs_v2/ddp_ours/model_ELU_UNet_gpu_4,5,6,7_ds_renoir_v2_nind_rid2021_v2_td_sidd_ps_128_bs_8_ep_150_lr_ad_0.001_lr_min_ad_0.0001_lam_ad_0.01_lam_kl_0.0_T_20.0_exp_id_v1/checkpoint/model_latest.pth' \
    --batch_size=8 \
    --lr_ad=1e-3 \
    --lr_min_ad=1e-4 \
    --lambda_ad=0.01 \
    --lambda_kl=0.0 \
    --temperature=20.0 \
    --exp_id='v1_continue' &
