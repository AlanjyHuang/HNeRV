"""
Evaluate patches at fractional positions (e.g., 0.5, 0.5 for middle of patches)
This tests the model's ability to generalize to positions between training patch centers
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image

from model_patch_dual import DualHeadHNeRV, PatchVideoDataSet, CLIPManager


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images (numpy arrays in [0,1] range)"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))


def calculate_ssim_torch(img1, img2, window_size=11):
    """
    Calculate SSIM using PyTorch
    img1, img2: torch tensors of shape [1, C, H, W] in [0, 1] range
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()
    
    # Create 2D Gaussian kernel
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(img1.size(1), 1, window_size, window_size).contiguous()
    window = window.to(img1.device)
    
    # Calculate SSIM
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
    
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def calculate_ssim(img1, img2):
    """Calculate SSIM between two numpy images"""
    # Convert to torch tensor [1, C, H, W]
    if img1.shape[0] == 3:  # CHW format
        img1_t = torch.from_numpy(img1).unsqueeze(0).float()
        img2_t = torch.from_numpy(img2).unsqueeze(0).float()
    else:  # HWC format
        img1_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()
        img2_t = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float()
    
    return calculate_ssim_torch(img1_t, img2_t)


def extract_ground_truth_patch(frame_tensor, norm_x, norm_y, patch_h, patch_w):
    """
    Extract ground truth patch at fractional position
    
    Args:
        frame_tensor: [3, H, W] tensor
        norm_x, norm_y: Normalized position [0, 1]
        patch_h, patch_w: Patch size in pixels
    
    Returns:
        Patch at the specified position [3, patch_h, patch_w]
    """
    _, H, W = frame_tensor.shape
    
    # Convert normalized coords to pixel coords (center of patch)
    center_x = norm_x * W
    center_y = norm_y * H
    
    # Calculate patch bounds
    x_start = int(center_x - patch_w / 2)
    y_start = int(center_y - patch_h / 2)
    x_end = x_start + patch_w
    y_end = y_start + patch_h
    
    # Clamp to valid range
    x_start = max(0, min(x_start, W - patch_w))
    y_start = max(0, min(y_start, H - patch_h))
    x_end = x_start + patch_w
    y_end = y_start + patch_h
    
    return frame_tensor[:, y_start:y_end, x_start:x_end]


def evaluate_fractional_positions(model, dataset, device, frame_indices, fractional_positions):
    """
    Evaluate model at fractional spatial positions
    
    Args:
        model: The model to evaluate
        dataset: The PatchVideoDataSet
        device: torch device
        frame_indices: List of frame indices to test
        fractional_positions: List of (norm_x, norm_y) tuples in [0, 1] range
    
    Returns:
        results: List of dicts with frame_idx, norm_x, norm_y, psnr, ssim, position_type
    """
    model.eval()
    results = []
    
    # Get patch dimensions from dataset
    patch_h = dataset.patch_h
    patch_w = dataset.patch_w
    
    num_frames = len(frame_indices)
    num_positions = len(fractional_positions)
    print("\nEvaluating {} frames at {} positions each".format(num_frames, num_positions))
    print("Patch size: {}x{}".format(patch_h, patch_w))
    
    with torch.no_grad():
        for frame_idx in tqdm(frame_indices, desc="Evaluating frames"):
            # Load full frame
            frame_tensor = dataset.img_transform(dataset.img_load(frame_idx))  # [3, H, W]
            
            # Normalized frame index
            norm_frame_idx = float(frame_idx) / len(dataset.video)
            
            for norm_x, norm_y, position_type in fractional_positions:
                # Create input coordinates [norm_frame_idx, norm_patch_x, norm_patch_y]
                input_coords = torch.tensor([[norm_frame_idx, norm_x, norm_y]], 
                                           dtype=torch.float32, device=device)
                
                # Forward pass
                rgb_output, clip_output, _, _ = model(input_coords)
                
                # Extract ground truth patch at this position
                gt_patch = extract_ground_truth_patch(frame_tensor, norm_x, norm_y, patch_h, patch_w)
                gt_patch = gt_patch.unsqueeze(0).to(device)  # [1, 3, H, W]
                
                # Resize model output if needed
                if rgb_output.shape[-2:] != gt_patch.shape[-2:]:
                    rgb_output = F.interpolate(
                        rgb_output, 
                        size=gt_patch.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Convert to numpy for metrics
                pred_patch = rgb_output[0].cpu().numpy()  # [3, H, W]
                target_np = gt_patch[0].cpu().numpy()  # [3, H, W]
                
                # Calculate metrics
                patch_psnr = calculate_psnr(target_np, pred_patch)
                patch_ssim = calculate_ssim(target_np, pred_patch)
                
                # Store results
                results.append({
                    'frame_idx': frame_idx,
                    'norm_x': norm_x,
                    'norm_y': norm_y,
                    'position_type': position_type,
                    'psnr': patch_psnr,
                    'ssim': patch_ssim
                })
    
    return results


def generate_test_positions(dataset):
    """
    Generate test positions including:
    1. Regular patch centers (trained positions)
    2. Middle positions between patches (0.5 offsets)
    3. Quarter positions (0.25, 0.75 offsets)
    
    Returns:
        List of (norm_x, norm_y, position_type) tuples
    """
    positions = []
    
    num_patches_h = dataset.num_patches_h
    num_patches_w = dataset.num_patches_w
    patch_h = dataset.patch_h
    patch_w = dataset.patch_w
    frame_height = dataset.frame_height
    frame_width = dataset.frame_width
    
    # 1. Regular patch centers (what the model was trained on)
    print("\nGenerating test positions:")
    print("  Grid: {}x{}".format(num_patches_h, num_patches_w))
    for row in range(num_patches_h):
        for col in range(num_patches_w):
            y_start = row * patch_h
            x_start = col * patch_w
            norm_x = (x_start + patch_w / 2) / frame_width
            norm_y = (y_start + patch_h / 2) / frame_height
            positions.append((norm_x, norm_y, 'center_{}_{}'.format(row, col)))
    
    print("  - {} patch centers (trained)".format(len(positions)))
    
    # 2. Middle positions between horizontally adjacent patches
    mid_positions_start = len(positions)
    for row in range(num_patches_h):
        for col in range(num_patches_w - 1):
            # Between patch (row, col) and (row, col+1)
            y_start = row * patch_h
            x_start = col * patch_w
            norm_x = (x_start + patch_w) / frame_width  # Right edge = left edge of next patch
            norm_y = (y_start + patch_h / 2) / frame_height
            positions.append((norm_x, norm_y, 'mid_h_{}_{}'.format(row, col)))
    
    # 3. Middle positions between vertically adjacent patches
    for row in range(num_patches_h - 1):
        for col in range(num_patches_w):
            # Between patch (row, col) and (row+1, col)
            y_start = row * patch_h
            x_start = col * patch_w
            norm_x = (x_start + patch_w / 2) / frame_width
            norm_y = (y_start + patch_h) / frame_height  # Bottom edge = top edge of next patch
            positions.append((norm_x, norm_y, 'mid_v_{}_{}'.format(row, col)))
    
    print("  - {} middle positions (untrained)".format(len(positions) - mid_positions_start))
    
    # 4. Corner positions (between 4 patches)
    corner_positions_start = len(positions)
    for row in range(num_patches_h - 1):
        for col in range(num_patches_w - 1):
            y_start = row * patch_h
            x_start = col * patch_w
            norm_x = (x_start + patch_w) / frame_width
            norm_y = (y_start + patch_h) / frame_height
            positions.append((norm_x, norm_y, 'corner_{}_{}'.format(row, col)))
    
    print("  - {} corner positions (untrained)".format(len(positions) - corner_positions_start))
    print("  Total: {} positions".format(len(positions)))
    
    return positions


def main():
    parser = argparse.ArgumentParser(description='Evaluate model at fractional patch positions')
    
    # Model checkpoint
    parser.add_argument('--weight', type=str, required=True, help='Path to model checkpoint')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/Kitchen', help='Path to video data')
    parser.add_argument('--vid', type=str, default='Kitchen', help='Video name')
    parser.add_argument('--crop_list', type=str, default='640_1280', help='Video crop size')
    parser.add_argument('--resize_list', type=str, default='-1', help='Video resize size')
    
    # Model parameters
    parser.add_argument('--embed', type=str, default='pe_1.25_80', help='Positional encoding')
    parser.add_argument('--fc_dim', type=int, default=192, help='FC dimension')
    parser.add_argument('--fc_hw', type=str, default='9_16', help='FC output size')
    parser.add_argument('--ks', type=str, default='0_3_3', help='Kernel sizes')
    parser.add_argument('--reduce', type=float, default=1.2, help='Channel reduction')
    parser.add_argument('--lower_width', type=int, default=32, help='Lowest channel width')
    parser.add_argument('--dec_strds', type=int, nargs='+', default=[5, 3, 2, 2, 2], help='Decoder strides')
    parser.add_argument('--num_blks', type=str, default='1_1', help='Block numbers')
    parser.add_argument('--conv_type', type=str, nargs='+', default=['convnext', 'pshuffel'], help='Conv types')
    parser.add_argument('--norm', type=str, default='none', help='Norm layer')
    parser.add_argument('--act', type=str, default='gelu', help='Activation')
    parser.add_argument('--out_bias', type=str, default='tanh', help='Output activation')
    parser.add_argument('--clip_dim', type=int, default=512, help='CLIP dimension')
    
    # Test parameters
    parser.add_argument('--num_frames', type=int, default=10, help='Number of frames to test')
    parser.add_argument('--frame_stride', type=int, default=1, help='Stride between test frames')
    
    # Output
    parser.add_argument('--out', type=str, default='output/fractional_eval', help='Output folder')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*80)
    print("FRACTIONAL PATCH POSITION EVALUATION")
    print("="*80)
    print("Device: {}".format(device))
    print("Model: {}".format(args.weight))
    print("Data: {}".format(args.data_path))
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Create dataset (without CLIP for speed)
    print("\n" + "="*80)
    print("Creating dataset...")
    print("="*80)
    
    # Temporarily disable data_split to get full dataset
    args.data_split = None
    dataset = PatchVideoDataSet(args)
    
    print("Video: {} frames".format(len(dataset.video)))
    print("Frame size: {}x{}".format(dataset.frame_height, dataset.frame_width))
    print("Patch grid: {}x{}".format(dataset.num_patches_h, dataset.num_patches_w))
    print("Patch size: {}x{}".format(dataset.patch_h, dataset.patch_w))
    
    # Load model
    print("\n" + "="*80)
    print("Loading model...")
    print("="*80)
    
    model = DualHeadHNeRV(args).to(device)
    checkpoint = torch.load(args.weight, map_location=device)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded from: {}".format(args.weight))
    
    # Generate test positions
    print("\n" + "="*80)
    print("Generating test positions...")
    print("="*80)
    
    test_positions = generate_test_positions(dataset)
    
    # Select frames to test
    total_frames = len(dataset.video)
    frame_indices = list(range(0, total_frames, args.frame_stride))[:args.num_frames]
    
    print("\nTesting {} frames: {}".format(len(frame_indices), frame_indices))
    
    # Evaluate
    print("\n" + "="*80)
    print("Evaluating...")
    print("="*80)
    
    results = evaluate_fractional_positions(model, dataset, device, frame_indices, test_positions)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save detailed results
    csv_path = os.path.join(args.out, 'fractional_positions_detailed.csv')
    df.to_csv(csv_path, index=False)
    print("\nDetailed results saved to: {}".format(csv_path))
    
    # Analyze results by position type
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Group by position type
    summary = df.groupby('position_type').agg({
        'psnr': ['mean', 'std', 'min', 'max'],
        'ssim': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print("\nPer-position-type statistics:")
    print(summary)
    
    # Overall statistics by category
    print("\n" + "="*80)
    print("CATEGORY COMPARISON")
    print("="*80)
    
    # Categorize positions
    df['category'] = df['position_type'].apply(lambda x: 
        'trained_center' if x.startswith('center_') else
        'untrained_middle_h' if x.startswith('mid_h_') else
        'untrained_middle_v' if x.startswith('mid_v_') else
        'untrained_corner' if x.startswith('corner_') else
        'other'
    )
    
    category_summary = df.groupby('category').agg({
        'psnr': ['mean', 'std'],
        'ssim': ['mean', 'std'],
        'position_type': 'count'
    }).round(4)
    category_summary.columns = ['PSNR_mean', 'PSNR_std', 'SSIM_mean', 'SSIM_std', 'count']
    
    print(category_summary)
    
    # Save category summary
    summary_path = os.path.join(args.out, 'fractional_positions_summary.csv')
    category_summary.to_csv(summary_path)
    print("\nSummary saved to: {}".format(summary_path))
    
    # Calculate degradation from trained centers
    trained_psnr = df[df['category'] == 'trained_center']['psnr'].mean()
    trained_ssim = df[df['category'] == 'trained_center']['ssim'].mean()
    
    print("\n" + "="*80)
    print("GENERALIZATION ANALYSIS")
    print("="*80)
    print("\nBaseline (Trained Centers):")
    print("  PSNR: {:.4f} dB".format(trained_psnr))
    print("  SSIM: {:.4f}".format(trained_ssim))
    
    for category in ['untrained_middle_h', 'untrained_middle_v', 'untrained_corner']:
        if category in df['category'].values:
            cat_df = df[df['category'] == category]
            cat_psnr = cat_df['psnr'].mean()
            cat_ssim = cat_df['ssim'].mean()
            psnr_drop = trained_psnr - cat_psnr
            ssim_drop = trained_ssim - cat_ssim
            
            print("\n{}:".format(category.replace('untrained_', '').replace('_', ' ').title()))
            print("  PSNR: {:.4f} dB (Δ {:+.4f} dB)".format(cat_psnr, psnr_drop))
            print("  SSIM: {:.4f} (Δ {:+.4f})".format(cat_ssim, ssim_drop))
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
