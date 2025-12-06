#!/bin/bash
# Direct comparison training: Patch-based vs Full-frame
# Uses same hyperparameters as the original train_nerv_clip.py for fair comparison

echo "=========================================="
echo "COMPARABLE Training Configuration"
echo "=========================================="
echo ""
echo "This matches the original full-frame training setup:"
echo "  • Model size: ~1.5M params (like original)"
echo "  • Data split: 9_10_10 (90% train, 10% val, like original)"
echo "  • Warmup: 50 epochs (same)"
echo "  • CLIP weight: 0.2 (same)"
echo "  • Epochs: 300 (same)"
echo ""
echo "Key difference:"
echo "  Original: Processes whole frames (640×1280)"
echo "  This: Processes 8 patches per frame (320×320 each)"
echo ""

CUDA_VISIBLE_DEVICES=7 python train_patch_dual.py \
    --data_path data/Kitchen \
    --vid Kitchen \
    --embed pe_1.25_80 \
    --fc_dim 96 \
    --fc_hw 9_16 \
    --dec_strds 5 2 2 \
    --batchSize 8 \
    --epochs 300 \
    --eval_freq 10 \
    --data_split 9_10_10 \
    --pixel_loss_warmup_epochs 50 \
    --clip_loss_weight 0.2 \
    --lr 0.001 \
    --lr_type cosine_0.1_1_0.1 \
    --loss Fusion6 \
    --norm none \
    --act gelu \
    --lower_width 12 \
    --reduce 1.5 \
    --clip_dim 512 \
    --outf output/Kitchen_patch_comparable

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Compare results:"
echo ""
echo "Original full-frame approach:"
echo "  • Location: output/clip_kitchen_new/"
echo "  • Method: Single network outputs whole frame + CLIP features"
echo "  • PSNR: Check their eval.txt"
echo ""
echo "Your patch-based approach:"
echo "  • Location: output/Kitchen_patch_comparable/"
echo "  • Method: Dual-head outputs RGB patch + CLIP embedding per patch"
echo "  • PSNR: Check rank0.txt or eval.txt"
echo ""
echo "Expected differences:"
echo "  • Patch PSNR might be slightly different (comparing patches vs whole frame)"
echo "  • CLIP similarity should be comparable if both methods work well"
echo "  • Model size: Both ~1.5M params (fair comparison)"
echo ""
