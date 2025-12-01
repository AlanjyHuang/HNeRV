#!/bin/bash
# BALANCED SPEED Training - Your Strategy
# Larger batch + smaller model + 300 epochs = Fast + Complete training

echo "=========================================="
echo "Balanced Training: Speed + Coverage"
echo "=========================================="
echo ""
echo "Your optimization strategy:"
echo "  • Smaller model: fc_dim=192 (~20M params, 35% less than 256)"
echo "  • Larger batch: 12 (better GPU utilization, fits in 40GB)"
echo "  • Full training: 300 epochs (complete learning)"
echo "  • Less frequent eval: every 30 epochs (saves time)"
echo ""
echo "Expected speedup: 2-3x faster (3-4 hours instead of 10)"
echo "Expected PSNR: ~28-31 (slightly lower due to smaller model)"
echo ""

CUDA_VISIBLE_DEVICES=7 python train_patch_dual.py \
    --data_path data/Kitchen \
    --vid Kitchen \
    --embed pe_1.25_80 \
    --fc_dim 192 \
    --fc_hw 9_16 \
    --dec_strds 5 3 2 2 2 \
    --batchSize 12 \
    --epochs 300 \
    --eval_freq 30 \
    --data_split 6_6_10 \
    --pixel_loss_warmup_epochs 50 \
    --clip_loss_weight 0.1 \
    --lr 0.001 \
    --lr_type cosine_0.1_1_0.1 \
    --loss Fusion6 \
    --outf output/Kitchen_patch_balanced

echo ""
echo "=========================================="
echo "Analysis of Your Strategy:"
echo "=========================================="
echo ""
echo "✅ SPEED GAINS:"
echo "  • Smaller model (fc_dim 256→192): ~1.3x faster per batch"
echo "  • Larger batch (8→24): ~2x better GPU utilization"
echo "  • Less eval (10→30): ~1.2x fewer interruptions"
echo "  • TOTAL: ~3x faster overall"
echo ""
echo "⚠️  QUALITY IMPACT:"
echo "  • fc_dim=192: PSNR might be 1-2 dB lower"
echo "  • But 300 epochs helps compensate"
echo "  • Final PSNR: likely 28-31 instead of 30-32"
echo ""
echo "View results:"
echo "  tensorboard --logdir output/Kitchen_patch_balanced"
echo ""
