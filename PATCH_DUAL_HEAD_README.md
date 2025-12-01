# Dual-Head Patch-Based HNeRV Model

## Overview

This is a modified HNeRV architecture that:
1. **Takes patch-based inputs**: Frame index (t) + patch position (x, y)
2. **Outputs dual heads**: 
   - RGB patch reconstruction
   - CLIP embedding for that patch
3. **Uses hybrid loss**: Pixel reconstruction loss + CLIP similarity loss
4. **Frame-based train/test split**: Some frames are used for training, others for validation

## Architecture

### Input
- `frame_idx` (normalized): Time coordinate [0, 1]
- `patch_x` (normalized): Horizontal patch position [0, 1]
- `patch_y` (normalized): Vertical patch position [0, 1]

### Model Structure
```
Input: (t, x, y) → Positional Encoding (PE)
                 ↓
             Shared Decoder
                 ↓
         ┌───────┴───────┐
         ↓               ↓
    RGB Head        CLIP Head
         ↓               ↓
    RGB Patch      512-dim CLIP
   (3×H×W)         embedding
```

### Key Features
1. **8 Patches per Frame**: Each frame is divided into a 2×4 grid (8 patches total)
2. **Positional Encoding**: Multi-dimensional PE for (t, x, y) coordinates
3. **Dual Heads**:
   - RGB head: 3×3 conv to output RGB patch
   - CLIP head: Global pooling + MLP to output 512-dim embedding
4. **Hybrid Loss**:
   - Pixel loss: L1/L2/SSIM for RGB reconstruction
   - CLIP loss: Cosine similarity with ground truth CLIP embeddings
   - Total loss = pixel_loss + λ * clip_loss

## Files

- `model_patch_dual.py`: Contains the dual-head model and patch-based dataset
  - `DualHeadHNeRV`: Main model class
  - `PatchVideoDataSet`: Dataset that extracts patches from frames
  - `CLIPManager`: Handles CLIP embedding extraction

- `train_patch_dual.py`: Training script with hybrid loss
  - Frame-based train/val split
  - Hybrid loss computation
  - Per-frame evaluation metrics

## Usage

### Training

```bash
python train_patch_dual.py \
    --data_path data/Kitchen \
    --vid Kitchen \
    --embed pe_1.25_80 \
    --fc_dim 512 \
    --fc_hw 9_16 \
    --dec_strds 5 3 2 2 2 \
    --num_blks 1_1 \
    --conv_type convnext pshuffel \
    --reduce 1.2 \
    --lower_width 32 \
    --crop_list 640_1280 \
    --data_split 18_19_20 \
    --epochs 300 \
    --batchSize 8 \
    --lr 0.001 \
    --lr_type cosine_0.1_1_0.1 \
    --loss L1 \
    --clip_loss_weight 0.1 \
    --pixel_loss_warmup_epochs 50 \
    --eval_freq 10 \
    --outf patch_dual_kitchen
```

### Key Arguments

#### Data Parameters
- `--data_path`: Path to video frames directory
- `--crop_list`: Crop size (e.g., `640_1280` for 640×1280)
- `--data_split`: Frame-based split ratio (e.g., `18_19_20` means 18/20 frames for training)

#### Architecture Parameters
- `--embed`: Positional encoding (e.g., `pe_1.25_80` for base=1.25, levels=80)
- `--fc_dim`: Feature dimension
- `--dec_strds`: Decoder strides (e.g., `5 3 2 2 2`)
- `--clip_dim`: CLIP embedding dimension (default: 512)

#### Loss Parameters
- `--loss`: Pixel loss type (`L1`, `L2`, `Fusion6`, etc.)
- `--clip_loss_weight`: Weight for CLIP loss (default: 0.1)
- `--pixel_loss_warmup_epochs`: Epochs before adding CLIP loss (default: 50)

### Evaluation

The model is evaluated per-frame by:
1. Reconstructing all 8 patches for each frame
2. Computing average PSNR across patches
3. Computing average CLIP similarity across patches
4. Separating metrics for train vs validation frames

Metrics reported:
- `train_psnr`: Average PSNR on training frames
- `val_psnr`: Average PSNR on validation frames
- `train_clip_sim`: Average CLIP similarity on training frames
- `val_clip_sim`: Average CLIP similarity on validation frames

## How It Works

### Dataset Processing

1. **Frame Loading**: Load video frames from directory
2. **Patch Extraction**: Divide each frame into 2×4 grid (8 patches)
3. **CLIP Embeddings**: Extract CLIP embeddings for each patch
4. **Coordinate Generation**: Generate (t, x, y) coordinates for each patch

### Training Process

1. **Input**: Sample batches of patches with coordinates
2. **Forward Pass**: 
   - Apply positional encoding to (t, x, y)
   - Pass through shared decoder
   - Split into RGB and CLIP heads
3. **Loss Computation**:
   - Pixel loss: Compare RGB output with ground truth patch
   - CLIP loss: Compare predicted embedding with ground truth
   - Warmup: Use only pixel loss for first N epochs
4. **Optimization**: Update model weights

### Frame-Based Split

Unlike patch-based splits, we split by frames:
- If frame is in training set → all 8 patches used for training
- If frame is in validation set → all 8 patches used for validation
- This ensures proper generalization testing to unseen frames

## Example Results

After training, you should see:
- Tensorboard logs in `output/<exp_id>/<param_str>/tensorboard/`
- Checkpoints in `output/<exp_id>/`
- Evaluation visualizations (if enabled) in `output/<exp_id>/visualize_patches/`

## Advantages

1. **Efficient Representation**: Learn compact representation for patches
2. **CLIP Alignment**: Embeddings align with CLIP's semantic space
3. **Flexible Resolution**: Can query any patch position continuously
4. **Semantic Awareness**: CLIP loss encourages semantically meaningful features

## Notes

- Patch size is automatically computed based on frame dimensions and grid size
- CLIP embeddings are cached to avoid redundant computation
- The model uses normalized coordinates [0, 1] for all inputs
- Positional encoding is applied to each coordinate dimension independently
