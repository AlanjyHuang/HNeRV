#!/bin/bash
# Evaluation script for patch-based dual-head model
# Comparable to the full-frame eval_only test

echo "=========================================="
echo "Evaluating Patch-Based Dual-Head Model"
echo "=========================================="
echo ""
echo "This evaluates a trained model and optionally dumps images"
echo "Similar to train_nerv_clip.py --eval_only"
echo ""

if [ -z "$1" ]; then
    echo "Usage: ./eval_patch_dual.sh <checkpoint_path> [--dump_images]"
    echo ""
    echo "Example:"
    echo "  ./eval_patch_dual.sh output/Kitchen_patch_balanced/model_best.pth"
    echo "  ./eval_patch_dual.sh output/Kitchen_patch_balanced/model_best.pth --dump_images"
    exit 1
fi

CHECKPOINT=$1
DUMP_IMAGES=""

if [ "$2" == "--dump_images" ]; then
    DUMP_IMAGES="--dump_images"
fi

echo "Checkpoint: $CHECKPOINT"
echo "Dump images: ${DUMP_IMAGES:-No}"
echo ""

CUDA_VISIBLE_DEVICES=7 python train_patch_dual.py \
    --data_path data/Kitchen \
    --vid Kitchen \
    --embed pe_1.25_80 \
    --fc_dim 192 \
    --fc_hw 9_16 \
    --dec_strds 5 3 2 2 2 \
    --batchSize 8 \
    --data_split 6_6_10 \
    --clip_dim 512 \
    --eval_only \
    --weight $CHECKPOINT \
    --eval_freq 1 \
    $DUMP_IMAGES \
    --outf output/eval_patch_dual

echo ""
echo "=========================================="
echo "Evaluation Results"
echo "=========================================="
echo ""
echo "Check output/eval_patch_dual/ for:"
echo "  • eval.txt - Quantitative metrics"
echo "  • eval.csv - Per-frame results"
if [ -n "$DUMP_IMAGES" ]; then
    echo "  • Reconstructed patch images"
fi
echo ""
echo "Compare with full-frame approach:"
echo "  Patch-based: Reconstructs 8 patches per frame"
echo "  Full-frame: Reconstructs entire frame at once"
echo ""
