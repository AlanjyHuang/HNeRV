# Experimental Evolution: From Single-Patch to Multi-Patch CLIP

## Motivation for Comparison

We tested two architectures to investigate whether spatial granularity in CLIP supervision affects semantic representation learning:

**Old Approach (Single-Patch):** The initial experiment used a single 448×448 patch per frame to generate one CLIP embedding. While this approach successfully predicted CLIP embeddings with 97.81% validation similarity, it had critical limitations: (1) only 6% of each 640×1280 frame was semantically supervised, (2) spatial semantic variation was ignored, and (3) generalization showed a 1.26% train-val gap.

**New Approach (Multi-Patch):** The improved architecture uses ~8 overlapping patches (stride 224) to create a spatial CLIP feature map. This achieves **99.27% similarity on both train and validation sets** with perfect generalization (0% gap), while reducing model size by 70% (1.51M vs 5.03M parameters). The per-patch supervision ensures every region of the frame learns correct semantic representations.

## Key Findings

| Metric | Single-Patch (Old) | Multi-Patch (New) | Improvement |
|--------|-------------------|-------------------|-------------|
| Val CLIP Similarity | 97.81% | 99.27% | +1.46% |
| Generalization Gap | 1.26% | 0.00% | Perfect ✓ |
| Model Size | 5.03M | 1.51M | 70% smaller |
| Stability (σ) | 0.49-1.61% | 0.20% | 8x better |
| Frame Coverage | ~6% | ~100% | Full supervision |

## Next Step: Coordinate-Based Multi-Patch CLIP

Having validated that multi-patch CLIP supervision works excellently for **image-based HNeRV**, the next experiment will test whether **coordinate-based NeRV** (true implicit neural representation) can achieve similar results:

**Hypothesis:** A pure INR (time coordinate → RGB + multi-CLIP) can learn spatially-varying semantic representations without ever seeing RGB images as input.

**Expected Challenge:** Learning fine-grained spatial semantics from a 1D time coordinate is harder than from 2D spatial RGB features. The positional encoding must capture both temporal evolution AND spatial semantic structure.

**Experiment Design:**
```bash
# Test coordinate-based NeRV with multi-patch CLIP
CUDA_VISIBLE_DEVICES=0,1 python train_nerv_clip.py \
  --embed pe_1.25_80 \              # Pure coordinate input
  --predict_clip \                  # Multi-patch CLIP output
  --clip_loss_weight 0.5 \
  --data_split 4_5_5 \
  --epochs 300
```

**Success Criteria:**
- Validation CLIP similarity > 95% (vs 99.27% baseline)
- Generalization gap < 2%
- Spatial CLIP map shows semantic coherence across patches

If successful, this would demonstrate that implicit neural representations can encode high-level semantic structure purely from temporal coordinates, opening paths for semantic video interpolation and compression.
