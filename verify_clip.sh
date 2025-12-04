#!/bin/bash

# Verify CLIP embeddings quality
# Tests if embeddings are semantically meaningful or just interpolated

CUDA_VISIBLE_DEVICES=7 python verify_clip_embeddings.py \
    --model_path output/1120/Kitchen/1_1_1__Dim64_16_FC9_16_KS0_1_5_RED1.2_low12_blk1_1_e300_b2_quant_M8_E6_lr0.001_cosine_0.1_1_0.1_L2_Size1.5_ENC_convnext_5,4,4,2,2_DEC_pshuffel_5,4,4,2,2_gelu1_1/model_best.pth \
    --data_path data/Kitchen \
    --vid Kitchen \
    --fc_dim 96 \
    --fc_hw 9_16 \
    --ks 0_3_3 \
    --num_blks 1_1 \
    --enc_strds 5 2 2 \
    --dec_strds 5 2 2 \
    --reduce 1.5 \
    --lower_width 12 \
    --conv_type convnext pshuffel \
    --norm none \
    --act gelu \
    --embed pe_1.25_80 \
    --clip_dim 512 \
    --out_bias tanh \
    --data_split 6_6_10 \
    --crop_list 640_1280 \
    --resize_list -1 \
    --patch_grid 2 4 \
    --max_samples 500 \
    --output clip_verification_results.json
