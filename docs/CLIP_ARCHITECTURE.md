# HNeRV with CLIP Embedding Prediction

## Architecture Overview

**Input:** (x, y, t) - Spatial-temporal coordinates
**Output:** RGB image + CLIP embeddings (512-dim)

## Model Flow

```
(x, y, t) 
   ↓
Positional Encoding (160 dims)
   ↓
Encoder (optional)
   ↓
Decoder Network
   ├→ RGB Head (3 channels) → RGB Image
   └→ CLIP Head (512 dims) → CLIP Embeddings
```

## CLIP Prediction Head

Added to HNeRV model in `model_all.py`:

```python
self.clip_head = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),  # Global pooling
    nn.Flatten(),
    nn.Linear(ngf, ngf * 2),
    nn.ReLU(inplace=True),
    nn.Linear(ngf * 2, clip_dim)  # Output 512-dim CLIP embedding
)
```

## Loss Function

**Hybrid Loss = Pixel Loss + λ * CLIP Loss**

- **Pixel Loss**: MSE/L1/Fusion6 between predicted and ground truth RGB
- **CLIP Loss**: `1 - cosine_similarity(pred_clip, gt_clip)`
- **λ** (clip_loss_weight): Default 0.1

## Training Strategy

1. **Warmup Phase** (epochs 0-50):
   - Train with **pixel loss only**
   - Lets model learn basic reconstruction
   
2. **CLIP Phase** (epochs 50+):
   - Add **CLIP loss** to encourage semantic preservation
   - Model learns to predict meaningful CLIP embeddings

## Usage

### Training with CLIP Prediction

```bash
python train_nerv_clip.py \
  --predict_clip \                    # Enable CLIP prediction head
  --clip_dim 512 \                    # CLIP embedding dimension
  --clip_loss_weight 0.1 \            # Weight for CLIP loss
  --pixel_loss_warmup_epochs 50 \     # Warmup before adding CLIP loss
  --data_path data/Kitchen \
  ... other args
```

### Training without CLIP (baseline)

```bash
python train_nerv_clip.py \
  --data_path data/Kitchen \
  ... other args
```

## Evaluation

The model is evaluated on:

1. **Pixel Metrics**: PSNR, MS-SSIM
2. **CLIP Similarity**: Cosine similarity between predicted and ground truth CLIP embeddings
   - Overall average
   - Train vs validation frames
   - Per-frame breakdown (saved to CSV)

### View Results

**TensorBoard:**
```bash
tensorboard --logdir output/your_experiment/*/tensorboard
```

Metrics tracked:
- `train/pixel_loss`
- `train/clip_loss`
- `train/total_loss`
- `eval/clip_similarity_all`
- `eval/clip_similarity_train`
- `eval/clip_similarity_val`
- `eval/clip_similarity_distribution` (histogram)

**Per-frame CSV:**
- `{outf}/clip_similarity_per_frame.csv`

### Verify CLIP Learning

Test if the model truly learned CLIP embeddings vs random:

```bash
python test_clip_learning.py \
  --checkpoint output/your_model/model_best.pth \
  --data_path data/Kitchen \
  --outf clip_verification
```

This compares your model against:
- Random embeddings baseline
- Shuffled frame-embedding pairs
- Temporal shuffle

## Key Differences from Previous Approach

| Aspect | Previous (Input-based) | Current (Prediction-based) |
|--------|------------------------|---------------------------|
| Input | (x,y,t) + CLIP (672 dims) | (x,y,t) only (160 dims) |
| Output | RGB only | RGB + CLIP |
| Learning | Implicit via input | Explicit via loss |
| Model Size | Larger (672-dim input) | Smaller (160-dim input) |
| Training Signal | Reconstruction only | Reconstruction + Semantic |

## Benefits

✅ Model learns to **predict semantic features** from coordinates
✅ Explicit CLIP loss ensures semantic preservation
✅ Can evaluate semantic quality during training
✅ Smaller input representation (160 vs 672 dims)
✅ More interpretable (can inspect predicted CLIP embeddings)

## Files Modified

- `model_all.py`: Added `clip_head` to HNeRV, modified forward() to return 4 outputs
- `train_nerv_clip.py`: Added CLIP loss computation, updated to handle 4-output model
- `test_clip_learning.py`: Verification script to test CLIP learning quality
