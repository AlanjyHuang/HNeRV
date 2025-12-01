# TensorBoard Quick Start Guide

## Starting TensorBoard

### Option 1: View all experiments
```bash
tensorboard --logdir output/
```

### Option 2: View specific experiment
```bash
tensorboard --logdir output/debug/test_warmup
```

### Option 3: View with specific port
```bash
tensorboard --logdir output/ --port 6007
```

## Accessing TensorBoard

After starting, open your browser to:
- **Default**: http://localhost:6006
- **Custom port**: http://localhost:PORT (replace PORT with your specified port)

## Key Metrics to Monitor

### 🔥 Warmup Verification

**Train/clip_loss_active**
- Location: SCALARS → Train
- What to see: Flat line at 0.0, then jumps to 1.0 at warmup epoch
- Purpose: Binary indicator of when CLIP loss is active

**Train/Loss/clip_loss**
- Location: SCALARS → Train/Loss
- What to see: Flat line at 0.0 during warmup, then increases
- Purpose: Shows CLIP loss value over time

**Train/Gradients/clip_head_norm**
- Location: SCALARS → Train/Gradients
- What to see: 0.0 during warmup, then non-zero after
- Purpose: Confirms CLIP head is not being trained during warmup

### 📊 Training Progress

**Train/PSNR**
- Location: SCALARS → Train
- What to see: Steadily increasing
- Purpose: RGB reconstruction quality

**Train/Loss/total_loss**
- Location: SCALARS → Train/Loss
- What to see: Decreasing over time, may jump slightly when CLIP activates
- Purpose: Overall training objective

**Train/learning_rate**
- Location: SCALARS → Train
- What to see: Follows your LR schedule (e.g., cosine decay)
- Purpose: Verify learning rate schedule

### 📈 Evaluation Metrics

**Eval/Frame/val_psnr**
- Location: SCALARS → Eval/Frame
- What to see: Should be > 0 and increasing (if you have validation frames)
- Purpose: Generalization quality

**Eval/Frame/val_clip_sim**
- Location: SCALARS → Eval/Frame
- What to see: Should be close to 1.0 (0.8+)
- Purpose: Semantic alignment on unseen frames

**Eval/Gap/psnr_gap**
- Location: SCALARS → Eval/Gap
- What to see: Small gap (<2-3 dB is good)
- Purpose: Measures overfitting

### 📉 Distributions

**Eval/Histogram/val_psnr_distribution**
- Location: DISTRIBUTIONS or HISTOGRAMS tab
- What to see: Distribution should be concentrated at high values
- Purpose: Understand variance in reconstruction quality

## TensorBoard Tips

### 1. Comparing Multiple Runs
- Check "Show data download links" in settings
- Use regex to select multiple runs: `test_warmup.*`
- Overlay multiple experiments to compare

### 2. Smoothing
- Use the smoothing slider (top left) to reduce noise
- Recommended: 0.6-0.8 for noisy metrics
- Set to 0 to see raw values

### 3. Custom Scalars
Create custom layouts:
1. Click "CUSTOM SCALARS" tab
2. Create layout with related metrics grouped together
3. Example layout:
```
Losses:
  - Train/Loss/pixel_loss
  - Train/Loss/clip_loss
  - Train/Loss/total_loss

Warmup:
  - Train/clip_loss_active
  - Train/Loss/clip_loss
  - Train/Gradients/clip_head_norm
```

### 4. Useful Filters
In the search box at the top:
- `Train/Loss` - Shows all training losses
- `Eval/Frame` - Shows all frame-level eval metrics
- `clip` - Shows all CLIP-related metrics
- `psnr` - Shows all PSNR metrics

### 5. Time Selection
- Click and drag on plot to zoom into time range
- Double-click to reset zoom
- Use "Horizontal Axis" dropdown to switch between:
  - STEP (epoch number)
  - RELATIVE (time since start)
  - WALL (absolute time)

## Keyboard Shortcuts

- `d` - Toggle dark mode
- `t` - Toggle tooltip
- `r` - Reload data
- `←/→` - Navigate between runs

## Troubleshooting

### "No dashboards are active"
- Make sure training has started and written some logs
- Check that the logdir path is correct
- Wait a few seconds and refresh

### Plots not updating
- Click the reload button (top right)
- Or close and restart TensorBoard

### Port already in use
```bash
# Kill existing TensorBoard process
pkill -f tensorboard

# Or use different port
tensorboard --logdir output/ --port 6007
```

### Can't access from browser
- Make sure you're using http:// not https://
- Try 127.0.0.1 instead of localhost
- Check firewall settings

## Example: Verifying Warmup in TensorBoard

1. **Start TensorBoard**:
   ```bash
   tensorboard --logdir output/debug/test_warmup
   ```

2. **Open browser**: http://localhost:6006

3. **Check SCALARS tab**:
   - Search for "clip_loss_active"
   - You should see: `Train/clip_loss_active`
   - Plot should show 0.0 for first epochs, then 1.0

4. **Add comparison**:
   - Also select `Train/Loss/clip_loss`
   - This should match: 0.0 when active=0, >0 when active=1

5. **Verify gradients**:
   - Search for "clip_head_norm"
   - Select `Train/Gradients/clip_head_norm`
   - Should be 0.0 during warmup

## Advanced: Remote TensorBoard

If training on a remote server:

1. **On remote server**:
   ```bash
   tensorboard --logdir output/ --port 6006 --bind_all
   ```

2. **SSH tunnel from local machine**:
   ```bash
   ssh -L 6006:localhost:6006 user@remote-server
   ```

3. **Open local browser**: http://localhost:6006

Or use VS Code's port forwarding feature if using Remote-SSH.
