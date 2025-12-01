#!/bin/bash
# Memory-efficient training script for patch-based dual-head model

echo "=========================================="
echo "Training Patch-Based Dual-Head Model"
echo "=========================================="
echo ""
echo "Memory-efficient configuration:"
echo "  • fc_dim: 128 (reduced from 512)"
echo "  • fc_hw: 5_8 (reduced from 9_16)"
echo "  • batch_size: 2 (reduced from 8)"
echo "  • data_split: 6_6_10 (60% train, 40% val, no waste)"
echo "  • This reduces first layer from 73K to 5K channels"
echo ""

CUDA_VISIBLE_DEVICES=7 python train_patch_dual.py \
    --data_path data/Kitchen \
    --vid Kitchen \
    --embed pe_1.25_80 \
    --fc_dim 128 \
    --fc_hw 5_8 \
    --batchSize 2 \
    --epochs 2 \
    --eval_freq 1 \
    --data_split 6_6_10 \
    --pixel_loss_warmup_epochs 50 \
    --clip_loss_weight 0.1 \
    --debug \
    --outf test_patch

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
```
#Split	Train%	Waste%	Val%	Pattern (every 10 frames)
#8_9_10	80%	10%	10%	0-7: train, 8: waste, 9: val
#6_6_10	60%	0%	40%	0-5: train, 6-9: val
#7_7_10	70%	0%	30%	0-6: train, 7-9: val
#8_8_10	80%	0%	20%	0-7: train, 8-9: val
```