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
echo "4. Use data_split 4_5_5 (80% train, 20% val, distributed pattern):"
echo "   → Frames 0-3: train, Frame 4: val, Frames 5-8: train, Frame 9: val, ..."
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
    --data_split 4_5_5 \
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
echo "=========================================="
echo "To view results in TensorBoard:"
echo "=========================================="
echo ""
echo "1. Start TensorBoard:"
echo "   tensorboard --logdir output/debug"
echo ""
echo "2. Open browser to: http://localhost:6006"
echo ""
echo "3. Key metrics to check:"
echo "   📊 SCALARS tab:"
echo "      • Train/clip_loss_active"
echo "        → Should be 0.0 for epochs 1-2"
echo "        → Should jump to 1.0 at epoch 3"
echo ""
echo "      • Train/Loss/clip_loss"
echo "        → Should be 0.0 for epochs 1-2"
echo "        → Should increase at epoch 3"
echo ""
echo "      • Train/Loss/total_loss"
echo "        → Should equal pixel_loss during warmup"
echo "        → Should increase when CLIP loss activates"
echo ""
echo "      • Train/Gradients/clip_head_norm"
echo "        → Should be 0.0 during warmup"
echo "        → Should become non-zero after warmup"
echo ""
echo "   📈 Compare plots:"
echo "      • Add both Train/Loss/pixel_loss and Train/Loss/clip_loss"
echo "      • CLIP loss should be flat at 0, then start changing"
echo ""
echo "4. Alternative: View specific experiment"
echo "   tensorboard --logdir output/debug/test_warmup"
echo ""
