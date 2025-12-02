#!/bin/bash
# Evaluate all patches (train + val) and compute per-patch PSNR/SSIM
# Outputs detailed CSV with frame_idx, patch_idx, psnr, ssim, split columns

echo "=========================================="
echo "Evaluating ALL Patches (Train + Val)"
echo "=========================================="
echo ""
echo "This evaluates every patch in both train and val sets"
echo "and computes PSNR/SSIM for each patch with train/val labels"
echo ""

if [ -z "$1" ]; then
    echo "Usage: ./eval_all_patches.sh <checkpoint_path>"
    echo ""
    echo "Example:"
    echo "  ./eval_all_patches.sh output/Kitchen_patch_balanced/model_best.pth"
    echo ""
    echo "This will evaluate:"
    echo "  - All train patches (6 frames × 8 patches = 48 patches for 6_6_10 split)"
    echo "  - All val patches (4 frames × 8 patches = 32 patches for 6_6_10 split)"
    echo ""
    echo "Output files in output/eval_all_patches/:"
    echo "  - all_patches_detailed.csv: Every patch with frame_idx, patch_idx, psnr, ssim, split"
    echo "  - per_frame_stats.csv: Aggregated stats per frame"
    echo "  - summary_stats.csv: Overall train/val statistics"
    exit 1
fi

CHECKPOINT=$1

echo "Checkpoint: $CHECKPOINT"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Run evaluation
# Note: Match these parameters to your trained model!
# For Kitchen_patch_comparable: fc_dim=96, dec_strds=5 2 2, reduce=1.5, lower_width=12
# IMPORTANT: Split 9_10_10 gives 0 val frames! Use 9_9_10 for 90% train, 10% val
CUDA_VISIBLE_DEVICES=7 python3 eval_all_patches.py \
    --weight $CHECKPOINT \
    --data_path data/Kitchen \
    --vid Kitchen \
    --embed pe_1.25_80 \
    --fc_dim 96 \
    --fc_hw 9_16 \
    --dec_strds 5 2 2 \
    --reduce 1.5 \
    --lower_width 12 \
    --data_split 9_9_10 \
    --clip_dim 512 \
    --out output/eval_all_patches

echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="
echo ""
echo "Output directory: output/eval_all_patches/"
echo ""
echo "Generated files:"
echo "  1. all_patches_detailed.csv"
echo "     - Columns: frame_idx, patch_idx, psnr, ssim, split"
echo "     - One row per patch"
echo "     - Use for detailed analysis of individual patches"
echo ""
echo "  2. per_frame_stats.csv"
echo "     - Columns: frame_idx, split, mean_psnr, min_psnr, max_psnr, std_psnr, ..."
echo "     - One row per frame (averaged across 8 patches)"
echo "     - Use to see which frames perform better/worse"
echo ""
echo "  3. summary_stats.csv"
echo "     - Overall statistics for train and val sets"
echo "     - Mean/std PSNR and SSIM for both splits"
echo ""
echo "Quick analysis commands:"
echo "  # View detailed results"
echo "  head -20 output/eval_all_patches/all_patches_detailed.csv"
echo ""
echo "  # Filter train patches only"
echo "  grep ',train' output/eval_all_patches/all_patches_detailed.csv"
echo ""
echo "  # Filter val patches only"
echo "  grep ',val' output/eval_all_patches/all_patches_detailed.csv"
echo ""
echo "  # See per-frame stats"
echo "  cat output/eval_all_patches/per_frame_stats.csv"
echo ""
