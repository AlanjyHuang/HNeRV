#!/bin/bash
# FAST Training Configuration - Optimized for Speed
# Reduces training time from 10+ hours to ~2-3 hours

echo "=========================================="
echo "FAST Training: Patch-Based Dual-Head Model"
echo "=========================================="
echo ""
echo "Speed Optimizations:"
echo "  • Fewer epochs: 100 (instead of 300)"
echo "  • Larger batch size: 16 (faster GPU utilization)"
echo "  • Less frequent eval: every 20 epochs"
echo "  • Warmup at epoch 20 (earlier activation)"
echo "  • Data: 60% train, 40% val"
echo ""
echo "Expected time: ~2-3 hours"
echo ""

CUDA_VISIBLE_DEVICES=4 python train_patch_dual.py \
    --data_path data/Kitchen \
    --vid Kitchen \
    --embed pe_1.25_80 \
    --fc_dim 256 \
    --fc_hw 9_16 \
    --dec_strds 5 3 2 2 2 \
    --batchSize 16 \
    --epochs 100 \
    --eval_freq 20 \
    --data_split 6_6_10 \
    --pixel_loss_warmup_epochs 20 \
    --clip_loss_weight 0.1 \
    --lr 0.001 \
    --lr_type cosine_0.1_1_0.1 \
    --loss Fusion6 \
    --outf output/Kitchen_patch_fast

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Next steps if PSNR is still ~30:"
echo "1. Try larger model (fc_dim=384)"
echo "2. Train longer (epochs=200)"
echo "3. Adjust learning rate"
echo ""
