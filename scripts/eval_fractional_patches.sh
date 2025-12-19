#!/bin/bash

# Evaluate model at fractional patch positions (e.g., 0.5, 0.5 between patches)
# This tests how well the model generalizes to positions not seen during training

# Default paths
WEIGHT="checkpoints/model_best.pth"
DATA_PATH="data/Kitchen"
VID="Kitchen"
OUTPUT="output/fractional_eval"

# Test parameters
NUM_FRAMES=20  # Number of frames to test
FRAME_STRIDE=10  # Test every 10th frame

echo "=========================================="
echo "Fractional Patch Position Evaluation"
echo "=========================================="
echo "This script evaluates the model at:"
echo "  1. Trained positions (patch centers)"
echo "  2. Middle positions between patches (0.5 offsets)"
echo "  3. Corner positions (intersections of 4 patches)"
echo ""
echo "Model: $WEIGHT"
echo "Data: $DATA_PATH"
echo "Testing $NUM_FRAMES frames (every ${FRAME_STRIDE}th frame)"
echo "=========================================="
echo ""

# Check if checkpoint exists
if [ ! -f "$WEIGHT" ]; then
    echo "Error: Model checkpoint not found at $WEIGHT"
    echo "Please specify the correct path with --weight argument"
    exit 1
fi

# Run evaluation
python core/eval_fractional_patches.py \
    --weight "$WEIGHT" \
    --data_path "$DATA_PATH" \
    --vid "$VID" \
    --num_frames "$NUM_FRAMES" \
    --frame_stride "$FRAME_STRIDE" \
    --out "$OUTPUT" \
    "$@"

echo ""
echo "=========================================="
echo "Results saved to: $OUTPUT"
echo "  - fractional_positions_detailed.csv (all measurements)"
echo "  - fractional_positions_summary.csv (category statistics)"
echo "=========================================="
