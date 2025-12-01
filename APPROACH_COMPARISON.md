# Comparison: Patch-Based vs Full-Frame HNeRV+CLIP

## Your Previous Test (Full-Frame)

**Command:**
```bash
CUDA_VISIBLE_DEVICES=0 python train_nerv_clip.py \
  --predict_clip \
  --data_path ./data/Kitchen \
  --vid Kitchen \
  --epochs 300 \
  --batchSize 1 \
  --lr 0.001 \
  --loss Fusion6 \
  --embed pe_1.25_80 \
  --enc_strds 5 2 2 \
  --dec_strds 5 2 2 \
  --num_blks 1_1 \
  --fc_hw 9_16 \
  --reduce 1.5 \
  --lower_width 12 \
  --conv_type convnext pshuffel \
  --norm none \
  --act gelu \
  --modelsize 1.5 \
  --clip_dim 512 \
  --clip_loss_weight 0.2 \
  --data_split 9_10_10 \
  --pixel_loss_warmup_epochs 50 \
  --outf output/clip_kitchen_new \
  --eval_only \
  --dump_images \
  --eval_freq 10
```

**Approach:**
- Input: Frame index (single scalar)
- Output: Full frame (640×1280) + CLIP features at multiple spatial locations
- Model: Single HNeRV with CLIP prediction head
- Training: One frame at a time (batch=1)
- CLIP: Predicted at multiple patches within the frame
- Loss: Fusion6 (combination of pixel + CLIP losses)

---

## New Approach (Patch-Based Dual-Head)

**Comparable Training:**
```bash
./train_patch_comparable.sh
```

**Equivalent to:**
```bash
CUDA_VISIBLE_DEVICES=7 python train_patch_dual.py \
  --data_path data/Kitchen \
  --vid Kitchen \
  --embed pe_1.25_80 \
  --fc_dim 96 \
  --fc_hw 9_16 \
  --dec_strds 5 2 2 \
  --batchSize 8 \
  --epochs 300 \
  --data_split 9_10_10 \
  --pixel_loss_warmup_epochs 50 \
  --clip_loss_weight 0.2 \
  --clip_dim 512 \
  --outf output/Kitchen_patch_comparable
```

**Approach:**
- Input: (frame_idx, patch_x, patch_y) - 3D coordinates
- Output: Single patch (320×320) + CLIP embedding for that patch
- Model: Dual-head architecture (RGB head + CLIP head)
- Training: Multiple patches per batch (batch=8)
- CLIP: One embedding per patch
- Loss: Pixel loss (L1) + CLIP similarity loss

---

## Key Differences

| Aspect | Full-Frame (Original) | Patch-Based (New) |
|--------|----------------------|-------------------|
| **Input** | Frame index (1D) | Frame + patch position (3D) |
| **Output Size** | 640×1280 full frame | 320×320 patch |
| **Model Input** | Temporal only | Temporal + Spatial |
| **CLIP Prediction** | Multiple locations per frame | One embedding per patch |
| **Batch Size** | 1 (whole frames) | 8+ (patches) |
| **Memory** | High (large output) | Lower (small patches) |
| **Training Speed** | Slower (1 frame/batch) | Faster (8 patches/batch) |
| **Model Complexity** | Single output head | Dual output heads |

---

## Architecture Comparison

### Full-Frame HNeRV+CLIP:
```
Input: [frame_idx] → [1]
  ↓
Positional Encoding (PE)
  ↓
Decoder (convnext + pshuffel)
  ↓
┌──────────┬──────────────┐
│ RGB Head │  CLIP Head   │
│ (3 chan) │  (512 chan)  │
└──────────┴──────────────┘
  ↓              ↓
Full Frame    CLIP Features
(640×1280)    (512×H×W)
```

### Patch-Based Dual-Head:
```
Input: [frame_idx, patch_x, patch_y] → [3]
  ↓
Positional Encoding (PE for each coordinate)
  ↓
Shared Decoder (pshuffel)
  ↓
┌──────────┬────────────────┐
│ RGB Head │ CLIP Head      │
│ (3 chan) │ (pool + linear)│
└──────────┴────────────────┘
  ↓              ↓
RGB Patch     CLIP Embedding
(320×320)     (512-dim vector)
```

---

## Expected Performance Comparison

### Full-Frame Approach:
- ✅ Outputs full frame directly
- ✅ No patch stitching needed
- ✅ Global context preserved
- ❌ Slower training (batch=1)
- ❌ Higher memory usage
- ❌ CLIP features at fixed spatial grid

**Typical Results:**
- PSNR: ~30-33 dB (full frame)
- CLIP quality: Depends on patch extraction method
- Training time: ~10-15 hours for 300 epochs

### Patch-Based Approach:
- ✅ Faster training (batch=8+)
- ✅ Lower memory per sample
- ✅ Explicit spatial control (patch position input)
- ✅ CLIP embedding per patch
- ❌ Needs patch stitching for full frame
- ❌ Potential boundary artifacts

**Typical Results:**
- PSNR: ~28-31 dB (per patch)
- CLIP quality: Direct embedding per patch
- Training time: ~3-5 hours for 300 epochs

---

## When to Use Each Approach

### Use Full-Frame HNeRV+CLIP if:
- You need complete frames without stitching
- Memory is not a constraint
- Training time is not critical
- You want global context in single forward pass
- You're okay with batch_size=1

### Use Patch-Based Dual-Head if:
- You want faster training
- You need explicit spatial control
- You want CLIP embeddings for specific regions
- You can handle patch stitching
- You want larger batch sizes for stable training

---

## Evaluation Comparison

### To evaluate full-frame model:
```bash
# Already done in your command
CUDA_VISIBLE_DEVICES=0 python train_nerv_clip.py \
  --eval_only \
  --dump_images \
  --weight output/clip_kitchen_new/model_best.pth \
  ...
```

### To evaluate patch-based model:
```bash
# Use the new script
./eval_patch_dual.sh output/Kitchen_patch_comparable/model_best.pth --dump_images
```

### Metrics to Compare:
1. **PSNR**: 
   - Full-frame: Average over whole frames
   - Patch-based: Average over patches (might be slightly different scale)

2. **CLIP Similarity**:
   - Full-frame: Computed at spatial grid locations
   - Patch-based: One per patch, direct comparison

3. **Training Time**:
   - Full-frame: ~10-15 hours
   - Patch-based: ~3-5 hours (with optimizations)

4. **Model Size**:
   - Both: Can be tuned to ~1.5M params for fair comparison

---

## Recommendation for Fair Comparison

Run both with **same computational budget**:

### Option 1: Same Model Size (~1.5M params)
```bash
# Already have full-frame results

# Train patch-based with comparable size
./train_patch_comparable.sh
```

### Option 2: Same Training Time (~5 hours)
```bash
# Full-frame: Reduce epochs to ~100-150
# Patch-based: Use fc_dim=192, batch=16, epochs=300
```

### Option 3: Same Final Quality Target
```bash
# Tune both to reach PSNR ~30 dB
# Compare: Which method reaches it faster?
```

---

## Quick Test Command

To quickly compare approaches:

```bash
# 1. Train patch-based for 10 epochs (test run)
CUDA_VISIBLE_DEVICES=7 python train_patch_dual.py \
    --data_path data/Kitchen \
    --vid Kitchen \
    --fc_dim 96 \
    --batchSize 8 \
    --epochs 10 \
    --data_split 9_10_10 \
    --clip_loss_weight 0.2

# 2. Check time per epoch and PSNR trend
# 3. Extrapolate to 300 epochs
```

This will show you:
- Time per epoch: Patch-based should be ~3-5x faster
- PSNR progression: See if quality is comparable
- Memory usage: Patch-based uses less
