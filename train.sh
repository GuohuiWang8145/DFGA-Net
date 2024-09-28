#DFGA-Net original model

#!/usr/bin/env bash
CHECKPOINT_DIR=output/Pre_Sceneflow && \
mkdir -p ${CHECKPOINT_DIR} && \
python main.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--no_resume_optimizer \
--stage sceneflow  \
--val_dataset things kitti15 \
--lr 4e-4 \
--minlr 1e-4 \
--num_scales 1 \
--batch_size 12 \
--accum_iter 2 \
--img_height 288 \
--img_width 512 \
--feature_channels 128 \
--padding_factor 16 \
--upsample_factor 8 \
--attn_type full \
--num_transformer_layers 6 \
--reg_refine 1 \
--num_reg_refine 3 \
--patience 10 \
--summary_freq 50 \
--val_freq 1 \
--save_ckpt_freq 1 \
--save_latest_ckpt_freq 1 \
--num_epochs 40


#!/usr/bin/env bash
CHECKPOINT_DIR=output/Ft-KITTImixed && \
mkdir -p ${CHECKPOINT_DIR} && \
python main.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume output/Pre_Sceneflow/pre-trained.pth \
--no_resume_optimizer \
--stage kitti15mix  \
--val_dataset kitti15 \
--lr 3e-4 \
--minlr 1e-5 \
--num_scales 1 \
--batch_size 12 \
--accum_iter 2 \
--img_height 288 \
--img_width 512 \
--feature_channels 128 \
--padding_factor 16 \
--upsample_factor 8 \
--attn_type full \
--num_transformer_layers 6 \
--reg_refine 1 \
--num_reg_refine 3 \
--patience 100 \
--summary_freq 50 \
--val_freq 1 \
--save_ckpt_freq 1 \
--save_latest_ckpt_freq 1 \
--num_epochs 500

#!/usr/bin/env bash
CHECKPOINT_DIR=output/Ft-ETH3D && \
mkdir -p ${CHECKPOINT_DIR} && \
python main.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume output/Pre_Sceneflow/pre-trained.pth \
--no_resume_optimizer \
--stage eth3d  \
--val_dataset kitti15 \
--lr 3e-4 \
--minlr 1e-5 \
--num_scales 1 \
--batch_size 12 \
--accum_iter 2 \
--img_height 288 \
--img_width 512 \
--feature_channels 128 \
--padding_factor 16 \
--upsample_factor 8 \
--inference_size 288 512 \
--attn_type full \
--num_transformer_layers 6 \
--reg_refine 1 \
--num_reg_refine 3 \
--patience 100 \
--summary_freq 50 \
--val_freq 1 \
--save_ckpt_freq 1 \
--save_latest_ckpt_freq 1 \
--num_epochs 200


#DFGA-Net-S model

#!/usr/bin/env bash
CHECKPOINT_DIR=output/Pre_Sceneflow && \
mkdir -p ${CHECKPOINT_DIR} && \
python main.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--no_resume_optimizer \
--stage sceneflow  \
--val_dataset things kitti15 \
--lr 4e-4 \
--minlr 1e-4 \
--num_scales 1 \
--batch_size 12 \
--accum_iter 2 \
--img_height 288 \
--img_width 512 \
--feature_channels 128 \
--padding_factor 16 \
--upsample_factor 8 \
--inference_size 352 1216 \
--attn_type swin \
--attn_splits_list 4 \
--num_transformer_layers 6 \
--reg_refine 1 \
--num_reg_refine 3 \
--patience 10 \
--summary_freq 50 \
--val_freq 1 \
--save_ckpt_freq 1 \
--save_latest_ckpt_freq 1 \
--num_epochs 40

#!/usr/bin/env bash
CHECKPOINT_DIR=output/Ft-KITTImixed && \
mkdir -p ${CHECKPOINT_DIR} && \
python main.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume output/Pre_Sceneflow/pre-trained.pth \
--no_resume_optimizer \
--stage kitti15mix  \
--val_dataset kitti15 \
--lr 1e-4 \
--minlr 1e-5 \
--num_scales 1 \
--batch_size 12 \
--accum_iter 2 \
--img_height 288 \
--img_width 512 \
--feature_channels 128 \
--padding_factor 16 \
--upsample_factor 8 \
--inference_size 352 1216 \
--attn_type swin \
--attn_splits_list 4 \
--num_transformer_layers 6 \
--reg_refine 1 \
--num_reg_refine 3 \
--patience 100 \
--summary_freq 50 \
--val_freq 1 \
--save_ckpt_freq 1 \
--save_latest_ckpt_freq 1 \
--num_epochs 500

#!/usr/bin/env bash
CHECKPOINT_DIR=output/Ft-ETH3D && \
mkdir -p ${CHECKPOINT_DIR} && \
python main.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume output/Pre_Sceneflow/pre-trained.pth \
--no_resume_optimizer \
--stage eth3d  \
--val_dataset kitti15 \
--lr 3e-4 \
--minlr 1e-5 \
--num_scales 1 \
--batch_size 12 \
--accum_iter 2 \
--img_height 288 \
--img_width 512 \
--feature_channels 128 \
--padding_factor 16 \
--upsample_factor 8 \
--inference_size 288 512 \
--attn_type swin \
--attn_splits_list 4 \
--num_transformer_layers 6 \
--reg_refine 1 \
--num_reg_refine 3 \
--patience 100 \
--summary_freq 50 \
--val_freq 1 \
--save_ckpt_freq 1 \
--save_latest_ckpt_freq 1 \
--num_epochs 200