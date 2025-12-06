#!/bin/bash

# Text-to-Patch Search Tool
# Search for video patches using natural language queries

# Example usage:
#   Interactive mode: bash search_patches.sh
#   Single query:     bash search_patches.sh --query "a silver refrigerator"

CUDA_VISIBLE_DEVICES=7 python search_patches_by_text.py \
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
    --top_k 10 \
    --output_dir search_results \
    --save_individual \
    --interactive \
    "$@"

echo ""
echo "Results saved to: search_results/"
