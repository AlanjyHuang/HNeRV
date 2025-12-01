#!/bin/bash
# QUALITY-FOCUSED Training - To push PSNR beyond 30
# Uses larger model and better training settings

echo "=========================================="
echo "HIGH QUALITY Training Configuration"
echo "=========================================="
echo ""
echo "Quality Improvements:"
echo "  • Larger model: fc_dim=384 (~65M params)"
echo "  • More decoder layers: dec_strds 5 3 2 2 2 2"
echo "  • Higher learning rate: 0.0015"
echo "  • L2 loss (better for high PSNR)"
echo "  • Larger batch for stability: 12"
echo ""
echo "Expected PSNR: 33-36 dB"
echo "Expected time: ~4-5 hours for 150 epochs"
echo ""

CUDA_VISIBLE_DEVICES=7 python train_patch_dual.py \
    --data_path data/Kitchen \
    --vid Kitchen \
    --embed pe_1.25_80 \
    --fc_dim 384 \
    --fc_hw 9_16 \
    --dec_strds 5 3 2 2 2 2 \
    --batchSize 12 \
    --epochs 150 \
    --eval_freq 10 \
    --data_split 6_6_10 \
    --pixel_loss_warmup_epochs 30 \
    --clip_loss_weight 0.1 \
    --lr 0.0015 \
    --lr_type cosine_0.1_1_0.1 \
    --loss L2 \
    --lower_width 48 \
    --reduce 1.15 \
    --outf output/Kitchen_patch_hq

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Check TensorBoard:"
echo "  tensorboard --logdir output/Kitchen_patch_hq"
echo ""
