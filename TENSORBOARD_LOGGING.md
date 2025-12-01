# TensorBoard Logging Guide

## Overview

The dual-head patch-based HNeRV model logs comprehensive metrics to TensorBoard for monitoring training progress and analyzing model behavior.

## How to View Logs

```bash
tensorboard --logdir output/patch_dual/<exp_id>/<param_str>/tensorboard/
```

## Logged Metrics

### 1. Training Metrics (Per Epoch)

#### Train/PSNR
- **Description**: Average PSNR across all training patches per epoch
- **Purpose**: Monitor RGB reconstruction quality on training data
- **Expected**: Should increase over time

#### Train/learning_rate
- **Description**: Current learning rate
- **Purpose**: Verify learning rate schedule is working correctly
- **Expected**: Should follow the specified schedule (e.g., cosine decay)

#### Train/Loss/pixel_loss
- **Description**: RGB reconstruction loss (L1/L2/Fusion)
- **Purpose**: Monitor pixel-level reconstruction error
- **Expected**: Should decrease over time

#### Train/Loss/clip_loss
- **Description**: CLIP similarity loss (1 - cosine_similarity)
- **Purpose**: Monitor semantic alignment with CLIP
- **Expected**: Should be 0 during warmup, then decrease after warmup

#### Train/Loss/total_loss
- **Description**: Combined loss = pixel_loss + λ × clip_loss
- **Purpose**: Overall training objective
- **Expected**: Should decrease over time

#### Train/Loss/clip_loss_weighted
- **Description**: Actual CLIP loss contribution (clip_loss × λ)
- **Purpose**: See the actual weight of CLIP loss in total loss
- **Expected**: Should be 0 during warmup

#### Train/Loss/pixel_to_clip_ratio
- **Description**: Ratio of pixel loss to CLIP loss
- **Purpose**: Balance between two loss components
- **Expected**: Shows relative importance of each loss

#### Train/clip_loss_active
- **Description**: Binary indicator (0 or 1)
- **Purpose**: Shows when CLIP loss is activated (after warmup)
- **Expected**: 0 for first N epochs, then 1

### 2. Gradient Metrics (Per Epoch)

#### Train/Gradients/total_norm
- **Description**: L2 norm of all gradients
- **Purpose**: Monitor gradient magnitude (detect vanishing/exploding gradients)
- **Expected**: Should be stable, not too small or too large

#### Train/Gradients/rgb_head_norm
- **Description**: L2 norm of RGB head gradients
- **Purpose**: Monitor RGB head training dynamics
- **Expected**: Should decrease as model converges

#### Train/Gradients/clip_head_norm
- **Description**: L2 norm of CLIP head gradients
- **Purpose**: Monitor CLIP head training dynamics
- **Expected**: Should be 0 during warmup, then stabilize

#### Train/Gradients/decoder_norm
- **Description**: L2 norm of shared decoder gradients
- **Purpose**: Monitor decoder training dynamics
- **Expected**: Should be stable

#### Train/Gradients/rgb_to_clip_ratio
- **Description**: Ratio of RGB head gradient norm to CLIP head gradient norm
- **Purpose**: Check if heads are learning at similar rates
- **Expected**: Should be balanced (not too skewed)

### 3. Batch-Level Metrics (Per Batch)

#### Train_Batch/PSNR
- **Description**: PSNR for current batch
- **Purpose**: Monitor training dynamics at fine granularity
- **Expected**: Noisy but trending upward

#### Train_Batch/pixel_loss
- **Description**: Pixel loss for current batch
- **Purpose**: Monitor loss dynamics during epoch
- **Expected**: Generally decreasing with some noise

#### Train_Batch/clip_loss
- **Description**: CLIP loss for current batch
- **Purpose**: Monitor CLIP loss dynamics
- **Expected**: 0 during warmup, then fluctuating

#### Train_Batch/total_loss
- **Description**: Total loss for current batch
- **Purpose**: Monitor overall training progress
- **Expected**: Decreasing trend

### 4. Model Parameters (Every 10 Epochs)

#### Model/Params/rgb_head_mean
- **Description**: Mean value of RGB head parameters
- **Purpose**: Detect parameter drift or saturation
- **Expected**: Should be stable

#### Model/Params/rgb_head_std
- **Description**: Standard deviation of RGB head parameters
- **Purpose**: Monitor parameter distribution
- **Expected**: Should be stable

#### Model/Params/clip_head_mean
- **Description**: Mean value of CLIP head parameters
- **Purpose**: Detect parameter drift or saturation
- **Expected**: Should be stable

#### Model/Params/clip_head_std
- **Description**: Standard deviation of CLIP head parameters
- **Purpose**: Monitor parameter distribution
- **Expected**: Should be stable

### 5. Evaluation Metrics (Per Eval)

#### Eval/Frame/train_psnr
- **Description**: Average PSNR on training frames (averaged across 8 patches per frame)
- **Purpose**: Monitor RGB quality on seen frames
- **Expected**: Should be high and increasing

#### Eval/Frame/val_psnr
- **Description**: Average PSNR on validation frames
- **Purpose**: Monitor RGB quality on unseen frames (generalization)
- **Expected**: Lower than train_psnr but should increase

#### Eval/Frame/train_clip_sim
- **Description**: Average CLIP similarity on training frames
- **Purpose**: Monitor semantic alignment on seen frames
- **Expected**: Should be high (close to 1.0)

#### Eval/Frame/val_clip_sim
- **Description**: Average CLIP similarity on validation frames
- **Purpose**: Monitor semantic alignment on unseen frames
- **Expected**: Should be high, may be slightly lower than train

#### Eval/Frame/train_clip_dist
- **Description**: CLIP distance (1 - similarity) on training frames
- **Purpose**: Alternative view of CLIP alignment
- **Expected**: Should be low (close to 0)

#### Eval/Frame/val_clip_dist
- **Description**: CLIP distance on validation frames
- **Purpose**: Alternative view of CLIP alignment
- **Expected**: Should be low

### 6. PSNR Statistics

#### Eval/PSNR_Stats/train_std
- **Description**: Standard deviation of PSNR across training frames
- **Purpose**: Measure consistency of reconstruction quality
- **Expected**: Lower is better (more consistent)

#### Eval/PSNR_Stats/val_std
- **Description**: Standard deviation of PSNR across validation frames
- **Purpose**: Measure consistency on unseen frames
- **Expected**: May be higher than train_std

#### Eval/PSNR_Stats/train_min
- **Description**: Minimum PSNR among training frames
- **Purpose**: Identify worst-case performance
- **Expected**: Should increase over time

#### Eval/PSNR_Stats/train_max
- **Description**: Maximum PSNR among training frames
- **Purpose**: Identify best-case performance
- **Expected**: Should be high

#### Eval/PSNR_Stats/val_min
- **Description**: Minimum PSNR among validation frames
- **Purpose**: Identify worst-case generalization
- **Expected**: Critical metric for robustness

#### Eval/PSNR_Stats/val_max
- **Description**: Maximum PSNR among validation frames
- **Purpose**: Identify best-case generalization
- **Expected**: Should be reasonable

### 7. Patch-Level Statistics

#### Eval/Patch/train_psnr_std
- **Description**: Standard deviation of PSNR across all training patches
- **Purpose**: Measure patch-level reconstruction consistency
- **Expected**: Indicates variation across spatial locations

#### Eval/Patch/val_psnr_std
- **Description**: Standard deviation of PSNR across all validation patches
- **Purpose**: Measure patch-level consistency on unseen data
- **Expected**: May be higher due to less training

#### Eval/Patch/train_clip_sim_std
- **Description**: Standard deviation of CLIP similarity across training patches
- **Purpose**: Measure semantic consistency across patches
- **Expected**: Lower means more consistent semantic understanding

#### Eval/Patch/val_clip_sim_std
- **Description**: Standard deviation of CLIP similarity across validation patches
- **Purpose**: Measure semantic consistency on unseen data
- **Expected**: Important for understanding generalization

### 8. Generalization Gap

#### Eval/Gap/psnr_gap
- **Description**: train_psnr - val_psnr
- **Purpose**: Measure overfitting in RGB reconstruction
- **Expected**: Small gap indicates good generalization

#### Eval/Gap/clip_sim_gap
- **Description**: train_clip_sim - val_clip_sim
- **Purpose**: Measure overfitting in semantic alignment
- **Expected**: Small gap indicates good generalization

### 9. Histograms

#### Eval/Histogram/train_psnr_distribution
- **Description**: Distribution of PSNR values across training frames
- **Purpose**: Visualize PSNR spread
- **Expected**: Should be concentrated at high values

#### Eval/Histogram/val_psnr_distribution
- **Description**: Distribution of PSNR values across validation frames
- **Purpose**: Visualize generalization quality distribution
- **Expected**: Compare with training distribution

#### Eval/Histogram/train_clip_sim_distribution
- **Description**: Distribution of CLIP similarity across training frames
- **Purpose**: Visualize semantic alignment spread
- **Expected**: Should be concentrated near 1.0

#### Eval/Histogram/val_clip_sim_distribution
- **Description**: Distribution of CLIP similarity across validation frames
- **Purpose**: Visualize semantic generalization
- **Expected**: Compare with training distribution

### 10. Dataset Information

#### Eval/Info/num_train_frames
- **Description**: Number of frames in training set
- **Purpose**: Verify data split
- **Expected**: Constant throughout training

#### Eval/Info/num_val_frames
- **Description**: Number of frames in validation set
- **Purpose**: Verify data split
- **Expected**: Constant throughout training

#### Eval/Info/num_train_patches
- **Description**: Total number of training patches (num_train_frames × 8)
- **Purpose**: Understand dataset size
- **Expected**: Constant = num_train_frames × 8

#### Eval/Info/num_val_patches
- **Description**: Total number of validation patches
- **Purpose**: Understand validation set size
- **Expected**: Constant = num_val_frames × 8

## Key Insights to Monitor

### 1. Training Progress
- **Primary**: `Train/PSNR`, `Train/Loss/total_loss`
- **Watch for**: Steady improvement, no sudden jumps

### 2. Loss Balance
- **Primary**: `Train/Loss/pixel_to_clip_ratio`, `Train/Loss/clip_loss_weighted`
- **Watch for**: Balanced contributions from both losses

### 3. Gradient Health
- **Primary**: `Train/Gradients/total_norm`
- **Watch for**: Stable values, not exploding or vanishing

### 4. Generalization
- **Primary**: `Eval/Gap/psnr_gap`, `Eval/Gap/clip_sim_gap`
- **Watch for**: Small gaps indicate good generalization

### 5. CLIP Alignment
- **Primary**: `Eval/Frame/val_clip_sim`
- **Watch for**: High values (>0.8) indicate good semantic learning

### 6. Consistency
- **Primary**: `Eval/PSNR_Stats/val_std`, `Eval/Patch/val_psnr_std`
- **Watch for**: Low values indicate consistent quality

## Example Training Curves

### Healthy Training:
1. `Train/PSNR`: Steadily increasing
2. `Train/Loss/total_loss`: Steadily decreasing
3. `Eval/Gap/psnr_gap`: Small and stable (<2-3 dB)
4. `Train/Gradients/total_norm`: Stable (0.1-10 range)
5. `Eval/Frame/val_clip_sim`: High (>0.85)

### Warning Signs:
1. **Overfitting**: Large `Eval/Gap/psnr_gap` (>5 dB)
2. **Gradient issues**: `Train/Gradients/total_norm` very small (<0.001) or very large (>100)
3. **Imbalanced training**: `Train/Gradients/rgb_to_clip_ratio` >> 10 or << 0.1
4. **Poor CLIP learning**: `Eval/Frame/val_clip_sim` < 0.7
5. **Loss instability**: `Train/Loss/total_loss` oscillating wildly

## Recommended TensorBoard Layout

Create custom TensorBoard layouts:

1. **Overview Tab**: 
   - Train/PSNR vs Eval/Frame/val_psnr
   - Train/Loss/total_loss
   - Train/learning_rate

2. **Loss Breakdown Tab**:
   - Train/Loss/pixel_loss
   - Train/Loss/clip_loss
   - Train/Loss/clip_loss_weighted
   - Train/clip_loss_active

3. **Gradients Tab**:
   - All Train/Gradients/* metrics

4. **Evaluation Tab**:
   - All Eval/Frame/* metrics
   - All Eval/Gap/* metrics

5. **Statistics Tab**:
   - All Eval/PSNR_Stats/* metrics
   - All Eval/Patch/* metrics

6. **Distributions Tab**:
   - All Eval/Histogram/* metrics
