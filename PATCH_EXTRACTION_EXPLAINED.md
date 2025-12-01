# How Patches Are Created - Detailed Explanation

## Overview
Patches are created **automatically** by the `PatchVideoDataSet` class in `model_patch_dual.py`. You don't need to do anything manually!

## Patch Grid Configuration

Currently set to **8 patches per frame** in a **2×4 grid**:

```python
# In PatchVideoDataSet.__init__()
self.num_patches_h = 2  # 2 rows
self.num_patches_w = 4  # 4 columns
self.num_patches = 8    # Total: 2 × 4 = 8 patches
```

## Visual Layout

For a **640×1280** Kitchen frame:

```
Original Frame (640×1280):
┌──────────────────────────────────────┐
│  Patch 0  │  Patch 1  │  Patch 2  │  Patch 3  │  ← Row 0
│  (320×320)│  (320×320)│  (320×320)│  (320×320)│    
├──────────────────────────────────────┤
│  Patch 4  │  Patch 5  │  Patch 6  │  Patch 7  │  ← Row 1
│  (320×320)│  (320×320)│  (320×320)│  (320×320)│
└──────────────────────────────────────┘
    Col 0      Col 1      Col 2      Col 3
```

**Each patch size:**
- Height: 640 ÷ 2 = **320 pixels**
- Width: 1280 ÷ 4 = **320 pixels**
- Result: **320×320** square patches

## How It Works (Step-by-Step)

### 1. Dataset Indexing
```python
def __len__(self):
    return len(self.video) * self.num_patches
    # Example: 192 frames × 8 patches = 1,536 total samples
```

### 2. Index to Frame+Patch Mapping
```python
def __getitem__(self, idx):
    frame_idx = idx // 8  # Which frame?
    patch_idx = idx % 8   # Which patch in that frame?
    
    # Examples:
    # idx=0  → frame=0, patch=0
    # idx=7  → frame=0, patch=7
    # idx=8  → frame=1, patch=0
    # idx=72 → frame=9, patch=0
```

### 3. Patch Position Calculation
```python
# Compute grid position
patch_row = patch_idx // 4  # Row in the grid (0 or 1)
patch_col = patch_idx % 4   # Column in the grid (0-3)

# Examples:
# patch_idx=0 → row=0, col=0 (top-left)
# patch_idx=3 → row=0, col=3 (top-right)
# patch_idx=4 → row=1, col=0 (bottom-left)
# patch_idx=7 → row=1, col=3 (bottom-right)
```

### 4. Extract Patch Pixels
```python
# Calculate pixel coordinates
y_start = patch_row * 320  # 0 or 320
y_end = y_start + 320      # 320 or 640
x_start = patch_col * 320  # 0, 320, 640, or 960
x_end = x_start + 320      # 320, 640, 960, or 1280

# Extract the patch
patch_img = full_frame[:, y_start:y_end, x_start:x_end]
# Shape: [3, 320, 320]
```

### 5. Create Input Coordinates
```python
# Normalize to [0, 1] range
norm_frame_idx = frame_idx / 192  # Time coordinate
norm_patch_x = (x_start + 160) / 1280  # X center of patch
norm_patch_y = (y_start + 160) / 640   # Y center of patch

input_coords = [norm_frame_idx, norm_patch_x, norm_patch_y]
# Example for patch 0 of frame 0:
# [0.0, 0.125, 0.25]  → top-left patch of first frame
```

### 6. Extract CLIP Embedding
```python
# Get CLIP embedding for this specific patch
clip_embed = clip_manager.get_patch_embeddings_grid(
    image_path, 
    num_patches_h=2, 
    num_patches_w=4
)[patch_idx]
# Returns: 512-dimensional embedding for this patch
```

## Output Format

Each sample from the dataset contains:

```python
{
    'img': torch.Tensor,          # Shape: [3, 320, 320] - RGB patch pixels
    'frame_idx': int,             # Which frame (0-191)
    'patch_idx': int,             # Which patch (0-7)
    'input_coords': torch.Tensor, # Shape: [3] - (t, x, y) normalized
    'clip_embed': torch.Tensor,   # Shape: [512] - CLIP embedding
}
```

## Customizing Patch Configuration

To change the number of patches, modify `PatchVideoDataSet.__init__()`:

### Example: 16 patches (4×4 grid)
```python
self.num_patches_h = 4  # 4 rows
self.num_patches_w = 4  # 4 columns
self.num_patches = 16   # Total: 4 × 4 = 16 patches
# Each patch: 160×320 (not square!)
```

### Example: 12 patches (3×4 grid)
```python
self.num_patches_h = 3  # 3 rows
self.num_patches_w = 4  # 4 columns
self.num_patches = 12   # Total: 3 × 4 = 12 patches
# Each patch: 213×320 (approximately)
```

### Example: 32 patches (4×8 grid)
```python
self.num_patches_h = 4  # 4 rows
self.num_patches_w = 8  # 8 columns
self.num_patches = 32   # Total: 4 × 8 = 32 patches
# Each patch: 160×160 (square!)
```

## Important Notes

1. **Automatic Processing**: You don't create patches manually - the dataset does it automatically during training

2. **CLIP Embeddings**: Cached per frame, so extracting 8 embeddings only happens once per frame

3. **Coordinates**: The model receives normalized (t, x, y) coordinates as input, not the patch pixels directly

4. **Frame-based Split**: Train/val split is by frame, so all 8 patches from a frame are in the same split

5. **Batch Processing**: With batch_size=8, you might get all 8 patches from one frame, or patches from different frames - it's randomized

## Memory Efficiency

The dataset:
- ✅ **Doesn't store all patches in memory** - extracts them on-the-fly
- ✅ **Caches CLIP embeddings per frame** - extracts once, reuses 8 times
- ✅ **Lazy loading** - only loads frames when needed

## Current Configuration Summary

- **Frame size**: 640×1280
- **Patch grid**: 2×4 (8 patches)
- **Patch size**: 320×320 (square)
- **Total samples**: 192 frames × 8 patches = 1,536
- **Train samples**: ~1,228 (80%)
- **Val samples**: ~308 (20%)
