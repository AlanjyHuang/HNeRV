# HNeRV with CLIP Embedding Prediction

**Implicit Neural Representation for Video with Semantic CLIP Embeddings**

## Overview

This project extends the HNeRV (Hybrid Neural Representation for Videos) architecture to jointly learn both visual reconstruction and semantic CLIP embeddings from spatial-temporal coordinates. The model learns to encode both pixel-level appearance and high-level semantic meaning into a continuous implicit neural representation.

## Architecture

### Input → Output
```
(x, y, t) coordinates → RGB Image (3 channels) + Multi-Patch CLIP Embeddings
```

### Network Structure
```
Spatial-Temporal Coordinates (x, y, t)
          ↓
    Positional Encoding (160 dims)
          ↓
    Encoder (ConvNeXt)
          ↓
    Decoder Network (produces spatial features)
          ├──→ RGB Head → Reconstructed Frame (640 × 1280 × 3)
          └──→ CLIP Head → Per-Patch CLIP Embeddings (512 × H × W)
```

### Per-Patch CLIP Prediction

The model predicts **multiple CLIP embeddings per frame**, one for each spatial patch:

- **Patch Size**: 448×448 pixels with stride 224 (50% overlap)
- **Patches per Frame**: ~8 patches (2 rows × 4 columns for 640×1280 images)
- **Prediction Method**: 1×1 convolutions on decoder features → [batch, 512, H, W]
- **Supervision**: Each spatial location supervised by its corresponding patch's CLIP embedding

### CLIP Prediction Head
```python
# Per-patch prediction using spatial convolutions
nn.Sequential(
    nn.Conv2d(ngf, ngf * 2, 1, 1, 0),    # 1×1 conv to expand features
    nn.ReLU(inplace=True),
    nn.Conv2d(ngf * 2, 512, 1, 1, 0),    # Output: [batch, 512, H, W]
    nn.Normalize(dim=1)                   # L2 normalize each embedding
)
```

**Why Per-Patch?**
- ✅ **Spatial Correspondence**: Each image region gets its own semantic embedding
- ✅ **Better Coverage**: Overlapping patches ensure full frame coverage
- ✅ **Independent Supervision**: Each patch embedding is directly supervised
- ✅ **Preserves Spatial Structure**: Feature map maintains spatial relationships

## Key Features

✨ **Dual-Output Architecture**: Predicts both RGB pixels and CLIP embeddings from coordinates  
🎯 **Hybrid Loss Function**: Combines pixel reconstruction loss with semantic similarity loss  
📊 **Multi-Metric Evaluation**: Tracks PSNR, SSIM, and CLIP similarity (train/val/overall)  
🚀 **Efficient Inference**: Direct CLIP prediction without re-encoding (30-100+ FPS)  
📈 **TensorBoard Integration**: Real-time monitoring of all losses and metrics  
🔍 **Verification Tools**: Statistical tests to validate CLIP learning quality  

## Training Strategy

### Two-Phase Training

**Phase 1: Pixel Warmup** (Epochs 0-50)
- Loss: `L_pixel` only
- Goal: Learn basic visual reconstruction
- Establishes stable feature representations

**Phase 2: Hybrid Learning** (Epochs 50+)
- Loss: `L_total = L_pixel + λ * L_CLIP`
- Goal: Preserve both visual fidelity and semantic meaning
- Default λ = 0.2

### Loss Functions

**Pixel Loss** (Fusion6):
```python
L_pixel = MSE(predicted_rgb, ground_truth_rgb)
# Compares every pixel in the reconstructed frame
```

**CLIP Loss** (Per-Patch Cosine Similarity):
```python
# For each patch:
#   1. Map patch coordinates to feature map position
#   2. Extract predicted embedding at that position
#   3. Compute similarity with ground truth patch embedding

L_CLIP = mean([1 - cosine_similarity(pred_patch[i], gt_patch[i]) 
               for i in all_patches])
# Averages loss across ~8 patches per frame
```

**Total Loss**:
```python
L_total = L_pixel + λ * L_CLIP
# λ = clip_loss_weight (default 0.5)
```

## Why Both Outputs?

| Aspect | RGB Output | CLIP Output |
|--------|-----------|-------------|
| **Measures** | Pixel-level accuracy | Semantic preservation |
| **Captures** | Colors, textures, edges | Objects, scenes, concepts |
| **Metrics** | PSNR, SSIM | Cosine similarity |
| **Ensures** | Visual fidelity | Semantic understanding |

**Together**: The model learns a complete representation that maintains both appearance and meaning.

## Installation

```bash
# Clone repository
git clone https://github.com/AlanjyHuang/HNeRV.git
cd HNeRV

# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

## Usage

### Training with Per-Patch CLIP Prediction (Recommended)

**80/20 Train/Validation Split:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_nerv_clip.py \
  --data_path data/Kitchen \
  --data_split 4_5_5 \
  --enc_strds 5 4 4 2 2 \
  --dec_strds 5 4 4 2 2 \
  --conv_type convnext pshuffel \
  --fc_hw 9_16 \
  --enc_dim 64_16 \
  --ks 0_1_5 \
  --reduce 1.2 \
  --lower_width 12 \
  --num_blks 1_1 \
  --modelsize 1.5 \
  --act gelu \
  --epochs 300 \
  -b 2 \
  --lr 0.001 \
  --quant_model_bit 8 \
  --quant_embed_bit 6 \
  --outf output/1120/Kitchen \
  --predict_clip \
  --clip_loss_weight 0.5 \
  --pixel_loss_warmup_epochs 50 \
  --distributed
```

**Key Configuration:**
- `--data_split 4_5_5`: 80% training (4 out of 5 frames), 20% validation (1 out of 5)
- `--predict_clip`: Enable per-patch CLIP prediction
- `--clip_loss_weight 0.5`: Balance between pixel and semantic loss
- `--pixel_loss_warmup_epochs 50`: Train 50 epochs with pixel-only before adding CLIP loss
- `-b 2`: Batch size per GPU (automatically divided by number of GPUs)
- `--distributed`: Multi-GPU training with 8× A100 GPUs

**For Different Splits:**
```bash
# 90/10 split (training/validation)
--data_split 9_10_10

# 70/30 split
--data_split 7_10_10

# Full training (no validation)
--data_split 1_1_1
```

### Key Arguments

| Argument | Description | Default | Recommended |
|----------|-------------|---------|-------------|
| `--predict_clip` | Enable per-patch CLIP prediction | False | True |
| `--clip_dim` | CLIP embedding dimension | 512 | 512 |
| `--clip_loss_weight` | Weight for CLIP loss (λ) | 0.1 | 0.5 |
| `--pixel_loss_warmup_epochs` | Epochs before adding CLIP loss | 50 | 50 |
| `--data_split` | Train/val split (X_Y_Z format) | 1_1_1 | 4_5_5 (80/20) |

**Data Split Format:** `X_Y_Z`
- For every Z frames: first X used for training, frames X to Y-1 skipped, frames Y+ used for validation
- Example `4_5_5`: Every 5 frames → first 4 train, last 1 validation = 80/20 split

### Training Baseline (No CLIP)

```bash
python train_nerv_clip.py \
  --data_path ./data/Kitchen \
  --vid Kitchen \
  # ... other args (without --predict_clip)
```

## Monitoring Training

### TensorBoard

```bash
# Local
tensorboard --logdir output/clip_experiment

# Remote server
# On server:
tensorboard --logdir output/clip_experiment --port 6006

# On local machine:
ssh -L 6006:localhost:6006 user@server

# Open browser:
http://localhost:6006
```

**Metrics Tracked:**
- `train/pixel_loss` - Pixel reconstruction loss
- `train/clip_loss` - CLIP semantic loss
- `train/total_loss` - Combined loss
- `eval/clip_similarity_all` - Overall CLIP similarity
- `eval/clip_similarity_train` - Training frames CLIP similarity
- `eval/clip_similarity_val` - Validation frames CLIP similarity
- `eval/clip_similarity_distribution` - Per-frame similarity histogram

### Output Files

```
output/experiment_name/
├── model_best.pth              # Best PSNR checkpoint
├── model_latest.pth            # Latest checkpoint
├── rank0.txt                   # Training logs
├── clip_similarity_per_frame.csv  # Detailed per-frame results
└── Encoder_X.XXM_Decoder_X.XXM_Total_X.XXM/
    └── tensorboard/            # TensorBoard logs
```

## Evaluation

### Automatic Evaluation (During Training)

Runs every `eval_freq` epochs. Reports:
- **Pixel Metrics**: PSNR, MS-SSIM (seen/unseen frames)
- **CLIP Metrics**: Cosine similarity (all/train/validation)

### Standalone Evaluation

```bash
python train_nerv_clip.py \
  --eval_only \
  --predict_clip \
  --resume output/experiment/model_best.pth \
  --data_path ./data/Kitchen \
  # ... other args
```

### Verify CLIP Learning

Test if the model truly learned semantic features vs. random:

```bash
python test_clip_learning.py \
  --checkpoint output/experiment/model_best.pth \
  --data_path ./data/Kitchen \
  --embed pe_1.25_80 \
  --fc_hw 9_16 \
  --enc_strds 5 2 2 \
  --dec_strds 5 2 2 \
  --num_blks 1_1 \
  --outf clip_verification_results
```

**Output:**
```
=====================================================================
CLIP LEARNING VERIFICATION RESULTS
=====================================================================

Condition                       Mean         Std
---------------------------------------------------------------------
Learned (Your Model)           0.8532      0.0421
Random Embeddings              0.0123      0.0234
Shuffled Pairing               0.2341      0.0512
Temporal Shuffle               0.3456      0.0623
---------------------------------------------------------------------

IMPROVEMENT ANALYSIS
Improvement over random:         0.8409  ✓ Significant
Improvement over shuffled:       0.6191  ✓ Significant

Overall Quality: EXCELLENT
Model has learned very strong semantic representations!
```

## Expected Results

### CLIP Similarity Interpretation

| Score | Quality | Meaning |
|-------|---------|---------|
| > 0.9 | Excellent | Strong semantic preservation |
| 0.7-0.9 | Good | Captures most semantic features |
| 0.4-0.7 | Moderate | Some semantic information preserved |
| < 0.4 | Poor | Limited semantic learning |

### Train vs. Validation Gap

- **Small gap** (< 0.05): Good generalization
- **Medium gap** (0.05-0.15): Acceptable, some overfitting
- **Large gap** (> 0.15): Model overfitting to training frames

## Experiment Design

### Research Questions

1. **Can an INR learn semantic embeddings from coordinates?**
   - Hypothesis: Yes, with proper supervision
   - Test: CLIP similarity > random baseline

2. **Does dual supervision improve representation quality?**
   - Hypothesis: Pixel + CLIP loss preserves both appearance and meaning
   - Test: Compare PSNR and CLIP similarity jointly

3. **What's the trade-off between pixel and semantic quality?**
   - Hypothesis: Balancing λ optimizes both metrics
   - Test: Sweep clip_loss_weight ∈ [0, 0.1, 0.2, 0.5, 1.0]

### Ablation Studies

```bash
# Baseline: Pixel only
python train_nerv_clip.py --outf output/baseline

# CLIP prediction enabled
python train_nerv_clip.py --predict_clip --clip_loss_weight 0.0 --outf output/clip_no_loss

# Different loss weights
python train_nerv_clip.py --predict_clip --clip_loss_weight 0.1 --outf output/clip_0.1
python train_nerv_clip.py --predict_clip --clip_loss_weight 0.2 --outf output/clip_0.2
python train_nerv_clip.py --predict_clip --clip_loss_weight 0.5 --outf output/clip_0.5

# Different warmup periods
python train_nerv_clip.py --predict_clip --pixel_loss_warmup_epochs 0 --outf output/no_warmup
python train_nerv_clip.py --predict_clip --pixel_loss_warmup_epochs 100 --outf output/warmup_100
```

## Technical Details

### Model Size

With `--modelsize 1.5`:
- Encoder: ~0.31M parameters
- Decoder: ~1.19M parameters  
- CLIP Head: ~0.01M parameters (additional)
- **Total**: ~1.51M parameters

### Memory Requirements

- **Training**: ~8-12 GB GPU memory (batch size 1, 640×1280 resolution)
- **Inference**: ~4-6 GB GPU memory

### Speed

| Mode | FPS | Notes |
|------|-----|-------|
| Training | 5-10 | Depends on resolution |
| Inference (with CLIP) | 30-100 | Direct prediction |
| Inference (RGB only) | 100-500 | No CLIP head |

## Project Structure

```
HNeRV/
├── train_nerv_clip.py          # Main training script
├── model_all.py                # Model architectures (HNeRV with CLIP head)
├── test_clip_learning.py       # CLIP learning verification
├── hnerv_utils.py              # Utility functions
├── efficient_nvloader.py       # Data loading
├── CLIP_ARCHITECTURE.md        # Detailed architecture docs
├── data/                       # Video datasets
│   └── Kitchen/
└── output/                     # Experiment outputs
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{hnerv_clip2025,
  title={HNeRV with CLIP Embedding Prediction: Learning Semantic Video Representations},
  author={Your Name},
  year={2025},
  url={https://github.com/AlanjyHuang/HNeRV}
}
```

## License

[Specify your license here]

## Acknowledgments

- Original HNeRV architecture
- OpenAI CLIP for semantic embeddings
- PyTorch team

## Contact

For questions or issues, please open a GitHub issue or contact [your email].
