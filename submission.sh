#!/usr/bin/env bash

## kitti
CUDA_VISIBLE_DEVICES=0  python main.py \
--submission \
--val_dataset kitti15 \
--inference_size 352 1216 \
--output_path submission/kitti \
--resume pretrained.pth \
--padding_factor 16 \
--upsample_factor 8 \
--num_scales 1 \
--num_head 1 \
--attn_type full \
--num_transformer_layers 6 \
--reg_refine \
--num_reg_refine 6

## eth3d
CUDA_VISIBLE_DEVICES=0  python main.py \
--submission \
--val_dataset eth3d \
--inference_size 352 1216 \
--output_path submission/kitti \
--resume pretrained.pth \
--padding_factor 16 \
--upsample_factor 8 \
--num_scales 1 \
--num_head 1 \
--attn_type full \
--num_transformer_layers 6 \
--eth_submission_mode test \
--reg_refine \
--num_reg_refine 6