#!/bin/bash
# Script to find maximum batch size that fits in GPU memory
# Your GPU: 40960 MiB available

echo "=========================================="
echo "Finding Maximum Batch Size"
echo "=========================================="
echo "GPU Memory: 40960 MiB"
echo ""

# Test with fc_dim=256 (full training config)
FC_DIM=256
FC_HW="9_16"
DEC_STRDS="5 3 2 2 2"

echo "Testing with full model config (fc_dim=$FC_DIM, fc_hw=$FC_HW)"
echo ""

for BATCH_SIZE in 2 4 8 16 32; do
    echo "----------------------------------------"
    echo "Testing batch_size=$BATCH_SIZE..."
    echo "----------------------------------------"
    
    CUDA_VISIBLE_DEVICES=7 python train_patch_dual.py \
        --data_path data/Kitchen \
        --vid Kitchen \
        --embed pe_1.25_80 \
        --fc_dim $FC_DIM \
        --fc_hw $FC_HW \
        --dec_strds $DEC_STRDS \
        --batchSize $BATCH_SIZE \
        --epochs 1 \
        --eval_freq 1 \
        --data_split 8_9_10 \
        --pixel_loss_warmup_epochs 50 \
        --clip_loss_weight 0.1 \
        --debug \
        --outf test_batch_size 2>&1 | head -100
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ batch_size=$BATCH_SIZE WORKS!"
        LAST_WORKING=$BATCH_SIZE
    else
        echo "❌ batch_size=$BATCH_SIZE FAILED (OOM)"
        echo ""
        echo "=========================================="
        echo "Maximum batch size found: $LAST_WORKING"
        echo "=========================================="
        echo ""
        echo "Recommended settings for full training:"
        echo "  --batchSize $LAST_WORKING"
        echo "  --fc_dim $FC_DIM"
        echo "  --fc_hw $FC_HW"
        exit 0
    fi
    
    echo ""
done

echo "=========================================="
echo "All tested batch sizes work!"
echo "Maximum tested: $LAST_WORKING"
echo "You can try even larger batch sizes if needed."
echo "=========================================="
