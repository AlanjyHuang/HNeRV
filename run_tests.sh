#!/bin/bash
# Simple test script for dual-head patch-based HNeRV model

echo "=========================================="
echo "Testing Dual-Head Patch-Based HNeRV Model"
echo "=========================================="
echo ""

# Test 1: Model architecture (no data needed)
echo "Test 1: Model Architecture"
echo "--------------------------"
python test_patch_dual.py
TEST1_STATUS=$?

if [ $TEST1_STATUS -eq 0 ]; then
    echo "✓ Model architecture test PASSED"
else
    echo "✗ Model architecture test FAILED"
    exit 1
fi

echo ""
echo "=========================================="
echo "All basic tests passed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Prepare your data in data/Kitchen/ directory"
echo "2. Run training with:"
echo "   python train_patch_dual.py --data_path data/Kitchen --vid Kitchen --debug"
echo "3. View logs with:"
echo "   tensorboard --logdir output/"
