#!/usr/bin/env bash

## Sceneflow
CUDA_VISIBLE_DEVICES=0  python main.py \
--eval \
--resume pth/DFE&FEwith6layer.pth \
--val_dataset things \
--num_scales 1 \
--num_head 1 \
--attn_type full \
--feature_channels 128 \
--padding_factor 16 \
--upsample_factor 8 \
--num_transformer_layers 6 \
--reg_refine 1 \
--num_reg_refine 6 \

## kitti
CUDA_VISIBLE_DEVICES=0  python main.py \
--eval \
--resume pth/DFE&FEwith6layer.pth \
--val_dataset kitti15 \
--num_scales 1 \
--num_head 1 \
--attn_type full \
--feature_channels 128 \
--padding_factor 16 \
--upsample_factor 8 \
--num_transformer_layers 6 \
--reg_refine 1 \
--num_reg_refine 6 \
--padding_factor 16 \
--upsample_factor 8 \
--feature_channels 128 \


## eth3d
CUDA_VISIBLE_DEVICES=0  python main.py \
--eval \
--resume pth/DFE&FEwith6layer.pth \
--val_dataset eth3d \
--num_scales 1 \
--num_head 1 \
--attn_type full \
--feature_channels 128 \
--padding_factor 16 \
--upsample_factor 8 \
--num_transformer_layers 6 \
--eth_submission_mode test \
--reg_refine 1 \
--num_reg_refine 6 \
--padding_factor 16 \
--upsample_factor 8 \
--feature_channels 128 \

