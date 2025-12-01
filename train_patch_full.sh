#!/bin/bash
# Full training script for patch-based dual-head HNeRV model
# This trains on the entire Kitchen dataset with proper configuration

echo "=========================================="
echo "Full Training: Patch-Based Dual-Head Model"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  • Dataset: Kitchen (640×1280 frames)"
echo "  • Patches: 8 patches per frame (2×4 grid)"
echo "  • Model: fc_dim=256 (~30M params - balanced capacity)"
echo "  • Training: 300 epochs with CLIP warmup at epoch 50"
echo "  • Data split: 6_6_10 (60% train, 40% val, no waste)"
echo "  • Batch size: 4"
echo "  • CLIP loss weight: 0.1"
echo ""
echo "Expected training time: Several hours"
echo ""

CUDA_VISIBLE_DEVICES=7 python train_patch_dual.py \
    --data_path data/Kitchen \
    --vid Kitchen \
    --embed pe_1.25_80 \
    --fc_dim 256 \
    --fc_hw 9_16 \
    --dec_strds 5 3 2 2 2 \
    --batchSize 4 \
    --epochs 300 \
    --eval_freq 10 \
    --data_split 6_6_10 \
    --pixel_loss_warmup_epochs 50 \
    --clip_loss_weight 0.1 \
    --lr 0.001 \
    --lr_type cosine_0.1_1_0.1 \
    --loss Fusion6 \
    --outf output/Kitchen_patch_dual_full

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "View results in TensorBoard:"
echo "  tensorboard --logdir output/Kitchen_patch_dual_full"
echo ""
echo "Key metrics to monitor:"
echo "  • Train/Loss/total_loss (should decrease)"
echo "  • Train/Loss/pixel_loss (should decrease)"
echo "  • Train/Loss/clip_loss (should activate after epoch 50)"
echo "  • Eval/PSNR (should increase)"
echo "  • Eval/CLIP_similarity (should increase after epoch 50)"
echo ""
