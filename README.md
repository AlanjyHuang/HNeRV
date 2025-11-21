# HNeRV with CLIP Embedding Prediction

**Implicit Neural Representation for Video with Semantic CLIP Embeddings**

## Overview

This project extends the HNeRV (Hybrid Neural Representation for Videos) architecture to jointly learn both visual reconstruction and semantic CLIP embeddings from spatial-temporal coordinates. The model learns to encode both pixel-level appearance and high-level semantic meaning into a continuous implicit neural representation.

## Architecture

### Input → Output
```
(x, y, t) coordinates → RGB Image (3 channels) + CLIP Embeddings (512 dimensions)
```

### Network Structure
```
Spatial-Temporal Coordinates (x, y, t)
          ↓
    Positional Encoding (160 dims)
          ↓
    Encoder (ConvNeXt)
          ↓
    Decoder Network
          ├──→ RGB Head → Reconstructed Frame (H × W × 3)
          └──→ CLIP Head → Semantic Embedding (512-dim)
```

### CLIP Prediction Head
```python
nn.Sequential(
    nn.AdaptiveAvgPool2d(1),      # Global spatial pooling
    nn.Flatten(),
    nn.Linear(ngf, ngf * 2),
    nn.ReLU(inplace=True),
    nn.Linear(ngf * 2, 512),      # Output CLIP embedding
    nn.Normalize()                 # L2 normalization
)
```

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
```

**CLIP Loss** (Cosine Similarity):
```python
L_CLIP = 1 - cosine_similarity(predicted_clip, ground_truth_clip)
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

### Training with CLIP Prediction

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
  --pixel_loss_warmup_epochs 50 \
  --outf output/clip_experiment
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--predict_clip` | Enable CLIP embedding prediction | False |
| `--clip_dim` | CLIP embedding dimension | 512 |
| `--clip_loss_weight` | Weight for CLIP loss (λ) | 0.1 |
| `--pixel_loss_warmup_epochs` | Epochs before adding CLIP loss | 50 |

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
