#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0  python main.py \
--eval \
--resume pth/DFE&FEwith9layer.pth \
--val_dataset things \
--num_scales 1 \
--num_head 1 \
--attn_type full \
--feature_channels 128 \
--padding_factor 16 \
--upsample_factor 8 \
--num_transformer_layers 9 \
--reg_refine 1 \
--num_reg_refine 3 \
--padding_factor 16 \
--upsample_factor 8 \
--feature_channels 128 \
