#!/bin/bash
# Test script to verify warmup mechanism works correctly

echo "=========================================="
echo "Testing CLIP Loss Warmup Mechanism"
echo "=========================================="
echo ""
echo "This test will:"
echo "1. Train for 3 epochs with warmup at epoch 2"
echo "2. Verify CLIP loss is 0.0000 during warmup"
echo "3. Verify CLIP loss becomes non-zero after warmup"
echo ""

CUDA_VISIBLE_DEVICES=7 python train_patch_dual.py \
    --data_path data/Kitchen \
    --vid Kitchen \
    --embed pe_1.25_80 \
    --fc_dim 128 \
    --dec_strds 5 3 2 \
    --batchSize 4 \
    --epochs 3 \
    --eval_freq 1 \
    --pixel_loss_warmup_epochs 2 \
    --clip_loss_weight 0.5 \
    --debug \
    --outf test_warmup

echo ""
echo "=========================================="
echo "Check the output above:"
echo "=========================================="
echo "✓ Epochs 1-2 should show: clip_loss:0.0000 [WARMUP]"
echo "✓ Epoch 3 should show: clip_loss:>0.0000 [CLIP_ACTIVE]"
echo "✓ Look for the '🔥 WARMUP COMPLETE!' message"
echo ""
echo "Or check TensorBoard:"
echo "  tensorboard --logdir output/debug/test_warmup"
echo "  Look at: Train/clip_loss_active (should be 0, then jump to 1)"
echo "           Train/Loss/clip_loss (should be 0, then increase)"
