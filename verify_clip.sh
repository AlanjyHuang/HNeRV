#!/bin/bash

# Verify CLIP embeddings quality
# Tests if embeddings are semantically meaningful or just interpolated

# Model: 9_10_10 split (90% train, 10% val)
# fc_dim inferred from model size, reduce=1.5, lower_width=12
# dec_strds: 5,2,2 (3 strides)
# CLIP weight: 0.2 (higher than previous 0.1)

CUDA_VISIBLE_DEVICES=7 python verify_clip_embeddings.py \
    --model_path output/output/Kitchen_patch_comparable/Kitchen/9_10_10_pe_1.25_80_FC9_16_KS0_3_3_RED1.5_low12_blk1_1_e300_b8_lr0.001_cosine_0.1_1_0.1_Fusion6_CLIP0.2_warmup50_DEC_pshuffel_5,2,2_gelu/epoch300.pth \
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
    --data_split 9_10_10 \
    --crop_list 640_1280 \
    --resize_list -1 \
    --patch_grid 2 4 \
    --max_samples 500 \
    --output clip_verification_results_9_10_10.json
