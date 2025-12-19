# Fractional Patch Position Evaluation

## Overview

This tool evaluates how well your HNeRV model generalizes to **patch positions it wasn't trained on**. During training, the model only sees discrete patch centers (e.g., patches 0-7 in a 2×4 grid). This evaluation tests what happens at fractional positions like the boundaries between patches.

## What It Tests

### 1. **Trained Positions** (Baseline)
Regular patch centers where the model was actually trained:
- Example: `(0.25, 0.16)` → center of patch (0, 1)
- These should give the best PSNR/SSIM scores

### 2. **Middle Positions** (Horizontal)
Positions halfway between horizontally adjacent patches:
- Example: `(0.5, 0.25)` → between patches (0, 1) and (0, 2)
- Tests horizontal interpolation capability

### 3. **Middle Positions** (Vertical)
Positions halfway between vertically adjacent patches:
- Example: `(0.25, 0.5)` → between patches (0, 0) and (1, 0)
- Tests vertical interpolation capability

### 4. **Corner Positions**
Intersections where 4 patches meet:
- Example: `(0.5, 0.5)` → intersection of patches (0,1), (0,2), (1,1), (1,2)
- Hardest positions, tests 2D interpolation

## Quick Start

```bash
# Using default settings (Kitchen video, 20 frames)
bash scripts/eval_fractional_patches.sh

# Custom checkpoint and more frames
bash scripts/eval_fractional_patches.sh \
    --weight checkpoints/my_model.pth \
    --num_frames 50 \
    --frame_stride 5
```

## Understanding the Results

### Output Files

**1. `fractional_positions_detailed.csv`**
- Every single measurement (frame × position)
- Columns: `frame_idx`, `norm_x`, `norm_y`, `position_type`, `psnr`, `ssim`, `category`

**2. `fractional_positions_summary.csv`**
- Statistics grouped by category
- Shows mean/std PSNR and SSIM for each category

### Key Metrics

The script prints a summary like:

```
CATEGORY COMPARISON
                     PSNR_mean  PSNR_std  SSIM_mean  SSIM_std  count
trained_center          35.42      2.14      0.9567     0.021    160
untrained_middle_h      33.78      2.56      0.9423     0.028    120
untrained_middle_v      33.91      2.48      0.9431     0.026    120
untrained_corner        32.15      2.89      0.9287     0.035     90

GENERALIZATION ANALYSIS

Baseline (Trained Centers):
  PSNR: 35.42 dB
  SSIM: 0.9567

Untrained Middle H:
  PSNR: 33.78 dB (Δ -1.64 dB)
  SSIM: 0.9423 (Δ -0.0144)
```

### What Good Results Look Like

- **Small degradation (< 1 dB PSNR drop)**: Model generalizes well, smoothly interpolates between patches
- **Moderate degradation (1-3 dB)**: Reasonable generalization, some boundary artifacts
- **Large degradation (> 3 dB)**: Model overfits to training positions, poor interpolation

## Why This Matters

### 1. **Continuous Representation**
Neural fields should be continuous functions. If performance drops sharply at untrained positions, the model is more like a lookup table than a true implicit function.

### 2. **Practical Applications**
- **Arbitrary-resolution decoding**: Generate video at different resolutions
- **Spatial super-resolution**: Sample between patches for finer details
- **Smooth panning**: Render positions smoothly without grid artifacts

### 3. **Model Quality Indicator**
Good generalization suggests:
- Effective positional encoding
- Sufficient model capacity
- Proper regularization (not memorizing positions)

## Coordinate System Explanation

### Normalized Coordinates [0, 1]

The model uses normalized spatial coordinates:

```
Frame (1280×640):
  num_patches_w = 4, num_patches_h = 2
  patch_w = 320, patch_h = 320

Patch Centers (trained):
  Patch (0,0): norm_x = 0.125, norm_y = 0.25
  Patch (0,1): norm_x = 0.375, norm_y = 0.25
  Patch (0,2): norm_x = 0.625, norm_y = 0.25
  Patch (0,3): norm_x = 0.875, norm_y = 0.25
  
  Patch (1,0): norm_x = 0.125, norm_y = 0.75
  Patch (1,1): norm_x = 0.375, norm_y = 0.75
  ...

Middle Positions (untrained):
  Between (0,0) and (0,1): norm_x = 0.25, norm_y = 0.25
  Between (0,1) and (0,2): norm_x = 0.50, norm_y = 0.25
  Between (0,0) and (1,0): norm_x = 0.125, norm_y = 0.50
  
Corner Positions (untrained):
  Intersection of (0,0), (0,1), (1,0), (1,1): norm_x = 0.25, norm_y = 0.50
```

## Example Use Cases

### Test Middle-of-Patch Positions Only

```python
# Modify eval_fractional_patches.py to only test specific positions
python core/eval_fractional_patches.py \
    --weight checkpoints/model_best.pth \
    --num_frames 10 \
    --out output/middle_only
```

### Visualize Specific Position

```python
import torch
from core.model_patch_dual import DualHeadHNeRV

# Load model
model = DualHeadHNeRV(args).cuda()
model.load_state_dict(torch.load('checkpoints/model_best.pth'))
model.eval()

# Test middle position (0.5, 0.5)
coords = torch.tensor([[0.5, 0.5, 0.5]]).cuda()  # [frame=0.5, x=0.5, y=0.5]
rgb_out, clip_out, _, _ = model(coords)

# Save image
import torchvision
torchvision.utils.save_image(rgb_out, 'middle_position.png')
```

## Advanced Analysis

### Compare Different Models

```bash
# Baseline model
bash scripts/eval_fractional_patches.sh \
    --weight checkpoints/baseline.pth \
    --out output/frac_baseline

# Your improved model
bash scripts/eval_fractional_patches.sh \
    --weight checkpoints/improved.pth \
    --out output/frac_improved

# Compare results
python analysis/compare_fractional_results.py \
    output/frac_baseline/fractional_positions_summary.csv \
    output/frac_improved/fractional_positions_summary.csv
```

### Test Different Offset Amounts

Modify `generate_test_positions()` in `eval_fractional_patches.py` to test positions at 0.25, 0.75 offsets instead of just 0.5:

```python
# Quarter positions (0.25 offset)
norm_x = (x_start + patch_w * 0.75) / frame_width
norm_y = (y_start + patch_h * 0.75) / frame_height
```

## Interpretation Guide

### High-Quality Neural Field (Good Generalization)
```
trained_center:      35.2 dB PSNR
untrained_middle:    34.5 dB PSNR  (Δ -0.7 dB)
untrained_corner:    34.1 dB PSNR  (Δ -1.1 dB)
→ Smooth, continuous representation
```

### Overfitted Model (Poor Generalization)
```
trained_center:      36.0 dB PSNR
untrained_middle:    31.2 dB PSNR  (Δ -4.8 dB)
untrained_corner:    28.5 dB PSNR  (Δ -7.5 dB)
→ Memorizing patch positions, not learning continuous function
```

### Potential Issues

1. **Large horizontal/vertical differences**: 
   - One direction generalizes worse than the other
   - May indicate anisotropic positional encoding

2. **Corner much worse than edges**:
   - 2D interpolation harder than 1D
   - Consider multi-scale features or better PE

3. **All positions similar**:
   - Good generalization, OR
   - Model has low capacity and produces blurry results everywhere

## Related Files

- `core/eval_fractional_patches.py` - Main evaluation script
- `scripts/eval_fractional_patches.sh` - Convenient wrapper
- `core/eval_all_patches.py` - Regular patch evaluation (trained positions only)
- `core/model_patch_dual.py` - Model definition with coordinate handling
