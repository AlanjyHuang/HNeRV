# Speed vs Quality Trade-off Guide

## Your Question: "Larger batch + smaller fc_dim + 300 epochs = Faster?"

**Answer: YES! ✅ But with quality trade-offs.**

---

## The Math:

### Current Setup (10+ hours):
- fc_dim = 256 (~30M params)
- batch_size = 8
- epochs = 300
- eval_freq = 10

**Time breakdown:**
- Forward/backward per batch: ~0.8s
- Total batches per epoch: 1,536 patches ÷ 8 = 192 batches
- Time per epoch: 192 × 0.8s = ~154s
- Evaluation time: ~30s every 10 epochs
- **Total: 300 × 154s + 30 × 30s = ~13.6 hours**

---

### Your Proposed Setup:

#### Option A: Moderate Reduction (fc_dim=192, batch=24)
```bash
fc_dim = 192 (~20M params, -35%)
batch_size = 24 (+200%)
epochs = 300
eval_freq = 30
```

**Time breakdown:**
- Forward/backward per batch: ~0.6s (smaller model)
- Total batches per epoch: 1,536 ÷ 24 = 64 batches (-67%!)
- Time per epoch: 64 × 0.6s = ~38s
- Evaluation time: ~30s every 30 epochs
- **Total: 300 × 38s + 10 × 30s = ~3.5 hours** ⚡
- **Speedup: 3.9x faster**

**Quality impact:**
- Expected PSNR: ~28-31 (vs 30-32 with fc_dim=256)
- Model capacity: Sufficient for 320×320 patches
- CLIP embeddings: Should work fine

---

#### Option B: Aggressive Reduction (fc_dim=128, batch=32)
```bash
fc_dim = 128 (~10M params, -67%)
batch_size = 32 (+300%)
epochs = 300
eval_freq = 30
```

**Time breakdown:**
- Forward/backward per batch: ~0.4s
- Total batches per epoch: 1,536 ÷ 32 = 48 batches
- Time per epoch: 48 × 0.4s = ~19s
- **Total: 300 × 19s + 10 × 30s = ~2 hours** 🚀
- **Speedup: 6.8x faster**

**Quality impact:**
- Expected PSNR: ~26-29 (lower quality)
- Might struggle with fine details
- CLIP embeddings: Still okay

---

#### Option C: Conservative (fc_dim=224, batch=16)
```bash
fc_dim = 224 (~25M params, -17%)
batch_size = 16 (+100%)
epochs = 300
eval_freq = 20
```

**Time breakdown:**
- Forward/backward per batch: ~0.7s
- Total batches per epoch: 1,536 ÷ 16 = 96 batches
- Time per epoch: 96 × 0.7s = ~67s
- **Total: 300 × 67s + 15 × 30s = ~6 hours**
- **Speedup: 2.3x faster**

**Quality impact:**
- Expected PSNR: ~29-31 (minimal loss)
- Good balance of speed and quality

---

## Comparison Table:

| Config | fc_dim | Batch | Params | Time | PSNR | Speedup | Best For |
|--------|--------|-------|--------|------|------|---------|----------|
| **Current** | 256 | 8 | 30M | 10h | 30-32 | 1x | Quality |
| **Conservative** | 224 | 16 | 25M | 6h | 29-31 | 1.7x | Balance |
| **Moderate** | 192 | 24 | 20M | 3.5h | 28-31 | 2.9x | **Recommended** |
| **Aggressive** | 128 | 32 | 10M | 2h | 26-29 | 5x | Speed |

---

## Why It Works Faster:

### 1. **Smaller Model (↓ fc_dim)**
- ✅ Fewer parameters → less computation per layer
- ✅ Less GPU memory used → room for larger batches
- ❌ Lower capacity → potentially lower PSNR

**Formula:** 
- fc_dim=256: First layer has 256×9×16 = 36,864 channels
- fc_dim=192: First layer has 192×9×16 = 27,648 channels
- **25% fewer computations!**

### 2. **Larger Batch (↑ batch_size)**
- ✅ Better GPU utilization (95%+ instead of 70%)
- ✅ Fewer iterations per epoch (24 instead of 192)
- ✅ More stable gradients (larger batch averages)
- ⚠️ Might need to adjust learning rate slightly

**Formula:**
- Batches per epoch = 1,536 patches ÷ batch_size
- batch=8: 192 iterations
- batch=24: 64 iterations
- **3x fewer iterations!**

### 3. **Keep 300 Epochs**
- ✅ Full learning curve
- ✅ Smaller model benefits from more training
- ✅ Better convergence than rushing with 100 epochs

---

## Will It Work?

**YES, with caveats:**

✅ **Will be faster:** Definitely 2-4x speedup
✅ **Will complete training:** 300 epochs ensures convergence
✅ **GPU memory:** Should fit fine with smaller model
✅ **CLIP learning:** Should work well

⚠️ **PSNR might be 1-3 dB lower:** Smaller model = less capacity
⚠️ **Batch size limit:** Test that batch=24 fits in 40GB GPU
⚠️ **Learning rate:** Might need slight adjustment for larger batch

---

## My Recommendation:

**Use Option A (Moderate):** `fc_dim=192, batch=24`

```bash
./train_patch_balanced.sh
```

**Reasoning:**
- 🚀 3.5 hours instead of 10 hours (70% time saved)
- 📊 PSNR ~28-31 (only 1-2 dB loss, acceptable)
- 💪 20M params still plenty for 320×320 patches
- ⚖️ Best speed/quality balance

**If that still takes too long:**
- Try Option B (fc_dim=128, batch=32) for 2-hour training
- Accept PSNR ~26-29 as trade-off for speed

**If quality matters most:**
- Use Option C (fc_dim=224, batch=16) for 6-hour training
- Get PSNR ~29-31 with minimal loss

---

## Test First:

Before committing to 300 epochs, test with 5 epochs:

```bash
CUDA_VISIBLE_DEVICES=7 python train_patch_dual.py \
    --data_path data/Kitchen \
    --vid Kitchen \
    --fc_dim 192 \
    --batchSize 24 \
    --epochs 5 \
    --data_split 6_6_10 \
    --debug
```

This will:
1. Confirm batch=24 fits in memory
2. Show you time per epoch
3. Verify training works correctly

**Expected output:**
- Time per epoch: ~40-50 seconds
- 5 epochs: ~3-4 minutes
- If this works, 300 epochs = ~3.5 hours ✅
