# train with multi dataset
#DDP
#sidd
#srun -p aipe --gres=gpu:4 --job-name='DGNet' -w SH-IDC2-172-20-21-216 \
#nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=10313 train_multiDataset_ddp_ours_encoder.py \
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

#srun -p aipe --gres=gpu:4 --job-name='DGNet' -w SH-IDC2-172-20-21-72 \
#nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=10320 train_multiDataset_ddp_ours_model.py \
#    --gpus=0,1,2,3 \
#    --data_set1='renoir_v2' \
#    --data_set2='nind' \
#    --data_set3='rid2021_v2' \
#    --data_set_test='sidd' \
#    --batch_size=8 \
#    --lr_ad=2e-4 \
#    --lr_min_ad=1e-5 \
#    --lambda_ad=0.001 \
#    --lambda_kl=0.0 \
#    --temperature=20.0 \
#    --exp_id='v1' &

#srun -p aipe --gres=gpu:4 --job-name='DGNet' -w SH-IDC2-172-20-21-72 \
#nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=10324 train_multiDataset_ddp_ours_model.py \
#    --gpus=4,5,6,7 \
#    --data_set1='renoir_v2' \
#    --data_set2='nind' \
#    --data_set3='rid2021_v2' \
#    --data_set_test='sidd' \
#    --batch_size=8 \
#    --lr_ad=2e-4 \
#    --lr_min_ad=1e-5 \
#    --lambda_ad=0.0001 \
#    --lambda_kl=0.0 \
#    --temperature=20.0 \
#    --exp_id='bn_2' &

#srun -p aipe --gres=gpu:4 --job-name='DGNet' -w SH-IDC2-172-20-21-216 \
#nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=10320 train_multiDataset_ddp_ours_model_patch.py \
#    --gpus=4,5,6,7 \
#    --data_set1='renoir_v2' \
#    --data_set2='nind' \
#    --data_set3='rid2021_v2' \
#    --data_set_test='sidd' \
#    --batch_size=8 \
#    --lr_ad=2e-4 \
#    --lr_min_ad=1e-5 \
#    --lambda_ad=0.0001 \
#    --lambda_kl=0.0 \
#    --temperature=20.0 \
#    --exp_id='patch_v3' &

srun -p aipe --gres=gpu:4 --job-name='DGNet' -w SH-IDC2-172-20-21-72 \
nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port=10319 train_multiDataset_ddp_ours_encoder.py \
    --gpus=0,1,2,3 \
    --data_set1='renoir_v2' \
    --data_set2='nind' \
    --data_set3='rid2021_v2' \
    --data_set_test='sidd' \
    --batch_size=8 \
    --lr_ad=2e-4 \
    --lr_min_ad=1e-5 \
    --lambda_ad=0.001 \
    --pretrained='/mnt/lustre/yangmingzhuo/DGNet/logs_v2/ddp_ours/model_ELU_UNet_gpu_0,1,2,3_ds_renoir_v2_nind_rid2021_v2_td_sidd_ps_128_bs_8_ep_150_lr_ad_0.0002_lr_min_ad_1e-05_lam_ad_0.001_lam_kl_0.0_T_20.0_exp_id_encoder_v3_target/checkpoint/model_latest.pth' \
    --lambda_kl=0 \
    --temperature=20.0 \
    --exp_id='encoder_v3_kl' &
