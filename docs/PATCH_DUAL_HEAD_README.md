# Dual-Head Patch-Based HNeRV Model

## Overview

This is a modified HNeRV architecture that:
1. **Takes patch-based inputs**: Frame index (t) + patch position (x, y)
2. **Outputs dual heads**: 
   - RGB patch reconstruction
   - CLIP embedding for that patch
3. **Uses hybrid loss**: Pixel reconstruction loss + CLIP similarity loss with warmup
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
4. **Hybrid Loss with Warmup**:
   - Pixel loss: L1/L2/Fusion6 (L1+SSIM) for RGB reconstruction
   - CLIP loss: Cosine similarity with ground truth CLIP embeddings
   - Warmup: Train with pixel loss only for first N epochs
   - Total loss = pixel_loss + λ * clip_loss (after warmup)

## Files

### Core Implementation
- **`model_patch_dual.py`**: Contains the dual-head model and patch-based dataset
  - `DualHeadHNeRV`: Main model class with dual outputs
  - `PatchVideoDataSet`: Dataset that extracts patches from frames
  - `CLIPManager`: Handles CLIP embedding extraction with caching

- **`train_patch_dual.py`**: Training script with hybrid loss and comprehensive logging
  - Frame-based train/val split
  - Hybrid loss computation with warmup
  - Per-frame evaluation metrics
  - TensorBoard logging for all metrics

### Training Scripts (Optimized Configurations)

- **`train_patch_comparable.sh`**: Fair comparison with original full-frame HNeRV
  - Model size: ~1.5M params
  - Loss: Fusion6 (70% L1 + 30% SSIM)
  - Best for: Direct comparison with original approach

- **`train_patch_balanced.sh`**: Speed-optimized configuration
  - fc_dim: 192 (~20M params)
  - batch_size: 12
  - Loss: Fusion6
  - Time: ~3-4 hours for 300 epochs
  - Best for: Fast training with good quality

- **`train_patch_fast.sh`**: Maximum speed
  - Epochs: 100 instead of 300
  - batch_size: 16
  - eval_freq: 20
  - Time: ~2-3 hours
  - Best for: Quick experiments

- **`train_patch_hq.sh`**: High quality configuration
  - fc_dim: 384 (~65M params)
  - Loss: L2 (highest PSNR)
  - More decoder layers
  - Time: ~4-5 hours for 150 epochs
  - Best for: Maximum quality

- **`train_patch_full.sh`**: Standard full training
  - fc_dim: 256 (~30M params)
  - Loss: Fusion6
  - 300 epochs
  - Best for: Balanced quality and training time

### Evaluation & Testing
- **`eval_patch_dual.sh`**: Evaluation script (like `--eval_only`)
- **`test_patch_dual.py`**: Quick architecture test
- **`test_warmup.sh`**: Verify warmup mechanism
- **`verify_warmup.py`**: Automated warmup verification

### Documentation
- **`PATCH_DUAL_HEAD_README.md`**: This file
- **`TENSORBOARD_LOGGING.md`**: Comprehensive logging documentation (40+ metrics)
- **`TENSORBOARD_QUICKSTART.md`**: Quick TensorBoard usage guide
- **`PATCH_EXTRACTION_EXPLAINED.md`**: Detailed explanation of patch creation
- **`APPROACH_COMPARISON.md`**: Comparison with full-frame HNeRV
- **`LOSS_FUNCTION_GUIDE.md`**: All available loss functions explained
- **`SPEED_OPTIMIZATION_GUIDE.md`**: Batch size and speed optimization tips

## Quick Start

### 1. Fast Training (Recommended First Run)
```bash
./train_patch_balanced.sh
```
- Time: ~3-4 hours
- Quality: PSNR ~28-31 dB
- Model: 20M params

### 2. Compare with Original
```bash
./train_patch_comparable.sh
```
- Same hyperparameters as original full-frame approach
- Fair comparison: both ~1.5M params

### 3. High Quality Training
```bash
./train_patch_hq.sh
```
- Larger model (65M params)
- Higher PSNR: ~33-36 dB
- Slower: ~4-5 hours

## Usage

### Training (Manual Configuration)

```bash
CUDA_VISIBLE_DEVICES=7 python train_patch_dual.py \
    --data_path data/Kitchen \
    --vid Kitchen \
    --embed pe_1.25_80 \
    --fc_dim 192 \
    --fc_hw 9_16 \
    --dec_strds 5 3 2 2 2 \
    --batchSize 12 \
    --epochs 300 \
    --eval_freq 30 \
    --data_split 6_6_10 \
    --pixel_loss_warmup_epochs 50 \
    --clip_loss_weight 0.1 \
    --lr 0.001 \
    --lr_type cosine_0.1_1_0.1 \
    --loss Fusion6 \
    --outf output/my_experiment
```

### Evaluation

```bash
./eval_patch_dual.sh output/my_experiment/model_best.pth --dump_images
```

### View Results in TensorBoard

```bash
tensorboard --logdir output/my_experiment
# Open browser to http://localhost:6006
```

## Key Arguments

### Data Parameters
- `--data_path`: Path to video frames directory
- `--crop_list`: Crop size (e.g., `640_1280` for 640×1280)
- `--data_split`: Frame-based split format `valid_train/total_train/total_data`
  - `6_6_10`: 60% train, 40% val (distributed pattern)
  - `8_9_10`: 80% train, 10% val, 10% unused
  - `9_10_10`: 90% train, 10% val

### Architecture Parameters
- `--embed`: Positional encoding (e.g., `pe_1.25_80` for base=1.25, levels=80)
- `--fc_dim`: Feature dimension (96=small, 192=medium, 256=standard, 384=large)
- `--fc_hw`: Initial spatial size (e.g., `9_16`)
- `--dec_strds`: Decoder strides (e.g., `5 3 2 2 2` for 5 layers)
- `--clip_dim`: CLIP embedding dimension (default: 512)
- `--reduce`: Channel reduction factor (default: 1.2)
- `--lower_width`: Minimum channel width (default: 32)

### Loss Parameters
- `--loss`: Pixel loss type
  - `L1`: Pure L1 loss (fast)
  - `L2`: Pure L2 loss (highest PSNR, can be blurry)
  - **`Fusion6`**: 70% L1 + 30% SSIM (**RECOMMENDED**, HNeRV standard)
  - `Fusion4`: 50% L1 + 50% SSIM (more perceptual)
  - See `LOSS_FUNCTION_GUIDE.md` for all options
- `--clip_loss_weight`: Weight for CLIP loss (0.1-0.2 recommended)
- `--pixel_loss_warmup_epochs`: Epochs before adding CLIP loss (50 recommended)

### Training Parameters
- `--batchSize`: Batch size (4-16 depending on GPU memory)
- `--epochs`: Number of training epochs (100-300)
- `--eval_freq`: Evaluate every N epochs
- `--lr`: Learning rate (0.001 standard)
- `--lr_type`: Learning rate schedule (`cosine_0.1_1_0.1` recommended)

### Debug & Logging
- `--debug`: Enable debug mode (process only 10-50 batches for quick testing)
- `--dump_images`: Save reconstructed images during evaluation
- `--eval_only`: Only evaluate, no training

## How It Works

### Dataset Processing

1. **Frame Loading**: Load video frames from directory (192 frames for Kitchen)
2. **Patch Extraction**: Divide each frame into 2×4 grid (8 patches of 320×320)
3. **CLIP Embeddings**: Extract CLIP embeddings for each patch (cached per frame)
4. **Coordinate Generation**: Generate normalized (t, x, y) coordinates for each patch
5. **Frame-based Split**: Split frames into train/val sets (all 8 patches from a frame stay together)

### Training Process

1. **Input**: Sample batches of patches with coordinates
2. **Forward Pass**: 
   - Apply positional encoding to (t, x, y) → 480-dim embedding
   - Pass through shared decoder (6 NeRV blocks)
   - Split into RGB head (3 channels) and CLIP head (512-dim)
3. **Loss Computation**:
   - Pixel loss: Fusion6 = 0.7×L1 + 0.3×SSIM between output and ground truth
   - CLIP loss: Cosine similarity between predicted and ground truth CLIP embeddings
   - Warmup: Use only pixel loss for first 50 epochs (CLIP loss = 0)
   - After warmup: Total loss = pixel_loss + 0.1 × clip_loss
4. **Optimization**: Adam optimizer with cosine learning rate schedule

### Evaluation

The model is evaluated per-frame by:
1. Reconstructing all 8 patches for each validation frame
2. Computing PSNR for each patch, average across frame
3. Computing CLIP similarity for each patch, average across frame
4. Separating metrics for train vs validation frames

Metrics reported:
- `Train_PSNR`: Average PSNR on training frames
- `Val_PSNR`: Average PSNR on validation frames  
- `Train_CLIP_sim`: Average CLIP similarity on training frames
- `Val_CLIP_sim`: Average CLIP similarity on validation frames

## Performance Expectations

### Training Time (Kitchen dataset, 192 frames)

| Configuration | fc_dim | Batch | Time (300 epochs) | PSNR | Quality |
|---------------|--------|-------|-------------------|------|---------|
| Fast | 192 | 16 | 2-3 hours (100 epochs) | ~28-30 | Good |
| Balanced | 192 | 12 | 3-4 hours | ~28-31 | Good |
| Standard | 256 | 8 | 6-8 hours | ~30-32 | Better |
| High Quality | 384 | 12 | 4-5 hours (150 epochs) | ~33-36 | Best |
| Comparable | 96 | 8 | 4-5 hours | ~26-29 | Small model |

### Memory Requirements

| fc_dim | Batch Size | GPU Memory | Max Batch (40GB GPU) |
|--------|------------|------------|----------------------|
| 96 | 8 | ~8 GB | 32 |
| 128 | 8 | ~10 GB | 24 |
| 192 | 8 | ~15 GB | 16 |
| 192 | 12 | ~22 GB | 12 |
| 256 | 8 | ~20 GB | 12 |
| 384 | 8 | ~30 GB | 8 |

**Note**: Actual memory usage depends on other processes on GPU. Monitor with `nvidia-smi`.

## Advantages over Full-Frame Approach

1. **Faster Training**: 2-3× speedup with larger batches (process 8-16 patches vs 1 frame)
2. **Lower Memory**: Process small patches (320×320) instead of full frames (640×1280)
3. **Explicit Spatial Control**: Input includes patch position (x, y)
4. **Direct CLIP Embeddings**: One 512-dim embedding per patch (cleaner than spatial CLIP features)
5. **Flexible**: Can query any patch position continuously

## Comparison with Original

### Original Full-Frame HNeRV+CLIP
- Input: Frame index only (1D)
- Output: Full frame + spatial CLIP features
- Batch size: 1 (memory limited)
- Training time: ~10-15 hours
- Model: Single head with CLIP prediction layer

### This Patch-Based Dual-Head
- Input: Frame index + patch position (3D)
- Output: Patch + CLIP embedding
- Batch size: 8-16 (faster)
- Training time: ~3-5 hours (optimized)
- Model: Dual heads (RGB + CLIP)

**Use patch-based for**: Speed, explicit spatial control, lower memory
**Use full-frame for**: Direct frame output, no stitching needed

## TensorBoard Metrics

The training script logs **40+ metrics** to TensorBoard:

### Loss Tracking
- Train/Loss/total_loss, pixel_loss, clip_loss
- Train/clip_loss_active (warmup indicator)
- Eval/Loss/pixel_loss, clip_loss

### Quality Metrics  
- Train/PSNR, Eval/PSNR (train vs val)
- Train/CLIP_similarity, Eval/CLIP_similarity
- Per-patch statistics

### Training Dynamics
- Train/Learning_rate
- Train/Gradients/* (gradient norms for each component)
- Train/Activations/* (activation distributions)

See `TENSORBOARD_LOGGING.md` for complete list and interpretation.

## Troubleshooting

### Out of Memory
- **Solution 1**: Reduce batch size (`--batchSize 4` or `--batchSize 8`)
- **Solution 2**: Use smaller model (`--fc_dim 128` instead of 192)
- **Solution 3**: Kill other processes on GPU (`nvidia-smi` to check)

### Validation Metrics Are 0
- **Cause**: Debug mode or data split issue
- **Solution 1**: Remove `--debug` flag to process all data
- **Solution 2**: Use data_split format like `6_6_10` or `8_9_10` (not `4_5_5`)

### Training Too Slow
- **Solution 1**: Increase batch size if memory allows
- **Solution 2**: Reduce eval frequency (`--eval_freq 30` instead of 10)
- **Solution 3**: Use fewer epochs (`--epochs 100`)
- **Solution 4**: Smaller model (`--fc_dim 128`)

### PSNR Stuck at ~30
- **Solution 1**: Use larger model (`--fc_dim 384`)
- **Solution 2**: Change loss to L2 or Fusion4
- **Solution 3**: Train longer (more epochs)
- **Solution 4**: Adjust learning rate

## Notes

- Patch size is automatically computed: frame_height÷2 × frame_width÷4
- CLIP embeddings are cached per frame to avoid redundant computation
- All coordinates are normalized to [0, 1] range
- Positional encoding is applied to each dimension independently
- Frame-based split ensures all patches from a frame stay in same set
- Warmup mechanism prevents CLIP loss from dominating early training
- Use Fusion6 loss (not pure L1) for best visual quality

## Citation

Based on HNeRV architecture:
```
@article{hnerv2023,
  title={HNeRV: A Hybrid Neural Representation for Videos},
  author={...},
  journal={CVPR},
  year={2023}
}
```
