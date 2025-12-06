# HNeRV with CLIP Embedding Prediction

**Patch-Based Dual-Head Neural Representation for Video with Semantic Understanding**

This project implements a dual-head patch-based HNeRV architecture that simultaneously learns RGB reconstruction and CLIP embeddings from spatial-temporal coordinates, enabling semantic video search and analysis.

---

## Overview

The model takes patch coordinates `(frame_idx, patch_x, patch_y)` as input and outputs:
1. RGB patch reconstruction (3×320×320)
2. CLIP semantic embedding (512-dim)

This enables both high-quality video compression and semantic understanding for applications like text-based video search.

---

## Project Structure

```
HNeRV/
├── core/                                # Core Implementation
│   ├── model_patch_dual.py              # DualHeadHNeRV architecture
│   ├── train_patch_dual.py              # Training script
│   ├── hnerv_utils.py                   # Utility functions
│   ├── search_patches_by_text.py        # Text-to-patch semantic search
│   ├── verify_clip_embeddings.py        # CLIP quality verification
│   ├── visualize_embedding_space.py     # t-SNE/UMAP visualization
│   ├── eval_all_patches.py              # Comprehensive evaluation
│   └── efficient_nvloader.py            # Optimized data loader
│
├── scripts/                             # Shell scripts
│   ├── train_patch_comparable.sh        # Main training (recommended)
│   ├── train_patch_fast.sh              # Quick experiments
│   ├── eval_all_patches.sh              # Evaluation
│   ├── search_patches.sh                # Interactive search
│   ├── verify_clip.sh                   # CLIP verification
│   └── visualize_embeddings.sh          # Embedding visualization
│
├── analysis/                            # Analysis tools
│   ├── plot_evaluation_results.py       # PSNR/SSIM/CLIP plots
│   ├── plot_training_logs.py            # Training curves
│   └── plot_clip_distribution.py        # Distribution analysis
│
├── tests/                               # Testing scripts
│   ├── test_patch_dual.py               # Dual-head tests
│   ├── test_clip_learning.py            # CLIP learning tests
│   └── verify_warmup.py                 # Warmup verification
│
├── legacy/                              # Legacy implementations
│   ├── model_all.py                     # Original HNeRV model
│   ├── train_nerv_all.py                # Original training
│   └── train_nerv_clip.py               # CLIP training v1
│
├── results/                             # Generated outputs
│   ├── embedding_space_comparison.png   # t-SNE visualization
│   └── *.csv                            # Evaluation data
│
├── docs/                                # Documentation
│   ├── PATCH_DUAL_HEAD_README.md        # Architecture details
│   ├── CLIP_ARCHITECTURE.md             # CLIP integration
│   ├── TENSORBOARD_QUICKSTART.md        # Monitoring guide
│   └── *.md                             # Other guides
│
├── data/                                # Video datasets
├── output/                              # Training checkpoints
└── checkpoints/                         # Pre-trained models
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/AlanjyHuang/HNeRV.git
cd HNeRV

# Install dependencies
pip install torch torchvision torchaudio
pip install git+https://github.com/openai/CLIP.git
pip install decord tensorboard scikit-learn umap-learn matplotlib pandas
```

### Prepare Data

```bash
# Place video frames in data/YourVideo/
# Format: 00000.png, 00001.png, 00002.png, ...
```

### Train

```bash
# Recommended: Comparable to baseline
bash scripts/train_patch_comparable.sh

# Alternatives:
# bash scripts/train_patch_fast.sh    (100 epochs, faster)
# bash scripts/train_patch_hq.sh      (larger model, best quality)
```

### Monitor

```bash
tensorboard --logdir output/
# Open http://localhost:6006
```

### Evaluate

```bash
# Comprehensive evaluation
bash scripts/eval_all_patches.sh output/.../epoch300.pth

# Text-based search
bash scripts/search_patches.sh

# Verify CLIP quality
bash scripts/verify_clip.sh

# Visualize embedding space
bash scripts/visualize_embeddings.sh
```

---

## Architecture

### Model Flow

```
Frame (640×1280) → 8 Patches (2×4 grid, each 320×320)
                   ↓
Input: (frame_idx, patch_x, patch_y) normalized [0,1]
                   ↓
         Positional Encoding
                   ↓
          Shared Decoder
                   ↓
        ┌──────────┴──────────┐
        ↓                     ↓
    RGB Head            CLIP Head
  (3×320×320)          (512-dim)
```

### Loss Function

```
Total Loss = Pixel Loss + λ × CLIP Loss

Where:
- Pixel Loss: Fusion6 (0.7×L1 + 0.3×SSIM)
- CLIP Loss: 1 - cosine_similarity(pred, gt)
- λ: 0.1-0.2 (activated after warmup)
```

### Training Strategy

**Phase 1 (Epochs 0-50)**: RGB reconstruction only (warmup)  
**Phase 2 (Epochs 50-300)**: Joint RGB + CLIP training  
**Data Split**: Frame-based (90% train, 10% validation)

---

## Results

### Performance Metrics (Kitchen Video, 192 frames)

| Data Split | Train PSNR | Val PSNR | Train CLIP | Val CLIP |
|------------|------------|----------|------------|----------|
| 9_10_10    | 31.61 dB   | -        | 0.9898     | -        |
| 6_6_10     | 32.61 dB   | 17.72 dB | 0.9897     | 0.9476   |

### CLIP Embedding Quality

**Verified Semantic Learning:**
- Ground truth similarity: 92.7% (embeddings are real, not random)
- Variance ratio: 69% (preserves semantic diversity)
- Temporal consistency: Smooth frame-to-frame transitions

**Known Issues:**
- RGB overfitting: 15 dB PSNR drop (train → val)
- CLIP overfitting: 4% similarity drop (train → val)
- Text search: Low similarity (~0.33, needs improvement)

### Visualization

![Embedding Space Comparison](results/embedding_space_comparison.png)

*t-SNE visualization showing model embeddings (compressed) vs ground truth (diverse)*

---

## Text-to-Patch Search

```bash
# Interactive search mode
bash scripts/search_patches.sh

# Example queries:
> a silver refrigerator
> coffee machine
> green plant on the counter
> kitchen sink
```

The search works for prominent objects with similarity scores around 0.33.

---

## Monitoring & Analysis

### TensorBoard Metrics

```bash
tensorboard --logdir output/
```

**Key metrics:**
- `Train/PSNR` - RGB reconstruction quality
- `Train/Loss/clip_loss` - CLIP alignment (0 during warmup)
- `Train/clip_loss_active` - Warmup indicator (0→1 at epoch 50)
- `Eval/CLIP_Similarity` - Validation performance

### Analysis Scripts

```bash
# Plot evaluation results
python analysis/plot_evaluation_results.py results/output.csv

# Plot training curves
python analysis/plot_training_logs.py output/.../rank0.txt

# CLIP similarity distribution
python analysis/plot_clip_distribution.py
```

---

## Advanced Usage

### Custom Training

```bash
python train_patch_dual.py \
    --data_path data/Kitchen \
    --vid Kitchen \
    --fc_dim 96 \
    --dec_strds 5 2 2 \
    --clip_loss_weight 0.2 \
    --pixel_loss_warmup_epochs 50 \
    --epochs 300 \
    --data_split 9_10_10
```

### Model Size Options

| Config | fc_dim | Params | PSNR | Speed |
|--------|--------|--------|------|-------|
| Small  | 64     | ~0.5M  | ~28  | Fast |
| Medium | 96     | ~1.5M  | ~31  | Balanced |
| Large  | 256    | ~30M   | ~33  | Slow |
| XL     | 384    | ~65M   | ~35  | Very Slow |

### Data Split Options

```
9_10_10 - 90% train, 10% val (recommended)
6_6_10  - 60% train, 40% val
8_8_10  - 80% train, 20% val
```

---

## Documentation

Comprehensive guides available in `docs/`:

- **PATCH_DUAL_HEAD_README.md** - Architecture overview
- **CLIP_ARCHITECTURE.md** - CLIP integration details
- **PATCH_EXTRACTION_EXPLAINED.md** - How patches work
- **LOSS_FUNCTION_GUIDE.md** - Loss function comparison
- **TENSORBOARD_QUICKSTART.md** - Monitoring guide
- **SPEED_OPTIMIZATION_GUIDE.md** - Training speedup tips
- **EXPERIMENT_EVOLUTION.md** - Design decisions and evolution

---

## Testing

### Run Tests

```bash
# Architecture tests
bash scripts/run_tests.sh

# CLIP verification (4 comprehensive tests)
bash scripts/verify_clip.sh
```

### Verification Tests

1. **Ground Truth Similarity** - Compare model vs real CLIP embeddings
2. **Embedding Diversity** - Check variance and pairwise distances
3. **Temporal Consistency** - Measure frame-to-frame changes
4. **Train/Val Difference** - Detect overfitting patterns

**Current Score:** 4/8 (embeddings are semantically meaningful but show overfitting)

---

## Citation

If you use this code, please cite:

```bibtex
@article{hnerv2023,
  title={HNeRV: A Hybrid Neural Representation for Videos},
  author={Chen, Hao and others},
  journal={CVPR},
  year={2023}
}
```

---

## License

This project is for research purposes. See original HNeRV license.

---

## Contributing

Contributions are welcome! Please:
- Open issues for bugs or questions
- Submit pull requests for improvements
- Share your experimental results

---

## Related Work

- [HNeRV](https://github.com/haochen-rye/HNeRV) - Original implementation
- [CLIP](https://github.com/openai/CLIP) - OpenAI's CLIP model
- [NeRF](https://www.matthewtancik.com/nerf) - Neural Radiance Fields

---

## Contact

**Author:** Alan Huang  
**Email:** ah212@rice.edu  
**GitHub:** [@AlanjyHuang](https://github.com/AlanjyHuang)

---

## Acknowledgments

- Chen et al. for the HNeRV architecture
- OpenAI for CLIP embeddings
- PyTorch team for the framework

---

**Status:** Active Research Project  
**Last Updated:** December 2025
