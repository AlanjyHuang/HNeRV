"""
Evaluate all patches across all frames and compute per-patch PSNR
Outputs detailed results showing train/val split with per-patch metrics
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


def evaluate_all_patches(model, dataset, device, split_name='all'):
    """
    Evaluate all patches and return per-patch metrics
    
    Returns:
        results: List of dicts with frame_idx, patch_idx, psnr, ssim, split
    """
    model.eval()
    results = []
    
    print(f"\nEvaluating {split_name} set: {len(dataset)} patches")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"Eval {split_name}"):
            # Get data
            patch_coords, target_patch, clip_target, frame_idx, patch_idx = dataset[idx]
            
            # Move to device
            patch_coords = patch_coords.unsqueeze(0).to(device)  # [1, N, 3]
            target_patch = target_patch.unsqueeze(0).to(device)  # [1, 3, H, W]
            
            # Forward pass
            rgb_output, clip_output = model(patch_coords)
            
            # Resize if needed
            if rgb_output.shape[-2:] != target_patch.shape[-2:]:
                rgb_output = nn.functional.interpolate(
                    rgb_output, 
                    size=target_patch.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Convert to numpy for metrics
            pred_patch = rgb_output[0].cpu().numpy()  # [3, H, W]
            target_np = target_patch[0].cpu().numpy()  # [3, H, W]
            
            # Calculate metrics
            patch_psnr = calculate_psnr(target_np, pred_patch)
            patch_ssim = calculate_ssim(target_np, pred_patch)
            
            # Store results
            results.append({
                'frame_idx': frame_idx,
                'patch_idx': patch_idx,
                'psnr': patch_psnr,
                'ssim': patch_ssim,
                'split': split_name
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate all patches with train/val split')
    
    # Model checkpoint
    parser.add_argument('--weight', type=str, required=True, help='Path to model checkpoint')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/Kitchen', help='Path to video data')
    parser.add_argument('--vid', type=str, default='Kitchen', help='Video name')
    parser.add_argument('--data_split', type=str, default='6_6_10', 
                        help='Train/val split (valid_train/total_train/total_data)')
    parser.add_argument('--crop_list', type=str, default='640_1280', help='Video crop size')
    
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
    parser.add_argument('--clip_dim', type=int, default=512, help='CLIP dimension')
    
    # Output
    parser.add_argument('--out', type=str, default='output/eval_all_patches', help='Output folder')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.outf, exist_ok=True)
    
    # Initialize CLIP manager
    clip_manager = CLIPManager(device=device)
    
    # Create datasets for train and val
    print("\n" + "="*80)
    print("Creating datasets...")
    print("="*80)
    
    train_dataset = PatchVideoDataSet(
        args.data_path,
        args.vid, 
        args.crop_list,
        args.data_split,
        clip_manager,
        is_train=True
    )
    
    val_dataset = PatchVideoDataSet(
        args.data_path,
        args.vid,
        args.crop_list, 
        args.data_split,
        clip_manager,
        is_train=False
    )
    
    print(f"\nTrain set: {len(train_dataset)} patches ({train_dataset.num_frames} frames × 8 patches)")
    print(f"Val set: {len(val_dataset)} patches ({val_dataset.num_frames} frames × 8 patches)")
    print(f"Total: {len(train_dataset) + len(val_dataset)} patches")
    
    # Build model
    print("\n" + "="*80)
    print("Building model...")
    print("="*80)
    
    model = DualHeadHNeRV(
        embed=args.embed,
        stem_dim_num=args.fc_dim,
        fc_hw_dim=args.fc_hw,
        expansion=args.reduce,
        num_blks=list(map(int, args.num_blks.split('_'))),
        norm=args.norm,
        act=args.act,
        bias=True,
        conv_type=args.conv_type,
        stride_list=args.dec_strds,
        clip_dim=args.clip_dim,
        lower_width=args.lower_width,
        ks=args.ks
    ).to(device)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.weight}")
    checkpoint = torch.load(args.weight, map_location=device)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.2f}M")
    
    # Evaluate train set
    print("\n" + "="*80)
    print("EVALUATING TRAIN SET")
    print("="*80)
    train_results = evaluate_all_patches(model, train_dataset, device, split_name='train')
    
    # Evaluate val set
    print("\n" + "="*80)
    print("EVALUATING VAL SET")
    print("="*80)
    val_results = evaluate_all_patches(model, val_dataset, device, split_name='val')
    
    # Combine results
    all_results = train_results + val_results
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save detailed results
    csv_path = os.path.join(args.outf, 'all_patches_detailed.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")
    
    # Compute statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Overall stats
    print("\nOVERALL:")
    print(f"  Mean PSNR: {df['psnr'].mean():.2f} dB")
    print(f"  Mean SSIM: {df['ssim'].mean():.4f}")
    
    # Train stats
    train_df = df[df['split'] == 'train']
    print("\nTRAIN SET:")
    print(f"  Patches: {len(train_df)}")
    print(f"  Mean PSNR: {train_df['psnr'].mean():.2f} dB")
    print(f"  Std PSNR: {train_df['psnr'].std():.2f} dB")
    print(f"  Mean SSIM: {train_df['ssim'].mean():.4f}")
    print(f"  Std SSIM: {train_df['ssim'].std():.4f}")
    
    # Val stats
    val_df = df[df['split'] == 'val']
    print("\nVAL SET:")
    print(f"  Patches: {len(val_df)}")
    print(f"  Mean PSNR: {val_df['psnr'].mean():.2f} dB")
    print(f"  Std PSNR: {val_df['psnr'].std():.2f} dB")
    print(f"  Mean SSIM: {val_df['ssim'].mean():.4f}")
    print(f"  Std SSIM: {val_df['ssim'].std():.4f}")
    
    # Per-patch position stats
    print("\nPER-PATCH POSITION (averaged across all frames):")
    print("-" * 60)
    for patch_idx in range(8):
        patch_df = df[df['patch_idx'] == patch_idx]
        print(f"  Patch {patch_idx}: PSNR={patch_df['psnr'].mean():.2f} dB, "
              f"SSIM={patch_df['ssim'].mean():.4f} "
              f"(train: {len(patch_df[patch_df['split']=='train'])}, "
              f"val: {len(patch_df[patch_df['split']=='val'])})")
    
    # Save summary statistics
    summary = {
        'overall_psnr': df['psnr'].mean(),
        'overall_ssim': df['ssim'].mean(),
        'train_psnr': train_df['psnr'].mean(),
        'train_ssim': train_df['ssim'].mean(),
        'train_psnr_std': train_df['psnr'].std(),
        'train_ssim_std': train_df['ssim'].std(),
        'val_psnr': val_df['psnr'].mean(),
        'val_ssim': val_df['ssim'].mean(),
        'val_psnr_std': val_df['psnr'].std(),
        'val_ssim_std': val_df['ssim'].std(),
        'train_patches': len(train_df),
        'val_patches': len(val_df),
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(args.outf, 'summary_stats.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary statistics saved to: {summary_path}")
    
    # Create per-frame summary
    print("\n" + "="*80)
    print("PER-FRAME STATISTICS")
    print("="*80)
    
    frame_stats = []
    for frame_idx in sorted(df['frame_idx'].unique()):
        frame_df = df[df['frame_idx'] == frame_idx]
        split = frame_df['split'].iloc[0]
        
        frame_stats.append({
            'frame_idx': frame_idx,
            'split': split,
            'mean_psnr': frame_df['psnr'].mean(),
            'min_psnr': frame_df['psnr'].min(),
            'max_psnr': frame_df['psnr'].max(),
            'std_psnr': frame_df['psnr'].std(),
            'mean_ssim': frame_df['ssim'].mean(),
            'min_ssim': frame_df['ssim'].min(),
            'max_ssim': frame_df['ssim'].max(),
            'std_ssim': frame_df['ssim'].std(),
        })
    
    frame_df = pd.DataFrame(frame_stats)
    frame_path = os.path.join(args.outf, 'per_frame_stats.csv')
    frame_df.to_csv(frame_path, index=False)
    print(f"Per-frame statistics saved to: {frame_path}")
    
    # Show sample of per-frame stats
    print("\nSample per-frame stats (first 5 train, first 5 val):")
    print("-" * 80)
    train_frames = frame_df[frame_df['split'] == 'train'].head()
    val_frames = frame_df[frame_df['split'] == 'val'].head()
    
    print("\nTrain frames:")
    for _, row in train_frames.iterrows():
        print(f"  Frame {row['frame_idx']:3d}: PSNR={row['mean_psnr']:.2f}±{row['std_psnr']:.2f} dB, "
              f"SSIM={row['mean_ssim']:.4f}±{row['std_ssim']:.4f}")
    
    print("\nVal frames:")
    for _, row in val_frames.iterrows():
        print(f"  Frame {row['frame_idx']:3d}: PSNR={row['mean_psnr']:.2f}±{row['std_psnr']:.2f} dB, "
              f"SSIM={row['mean_ssim']:.4f}±{row['std_ssim']:.4f}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.outf}/")
    print(f"  - all_patches_detailed.csv: Per-patch metrics for all patches")
    print(f"  - per_frame_stats.csv: Aggregated stats per frame")
    print(f"  - summary_stats.csv: Overall summary statistics")
    print("")


if __name__ == '__main__':
    main()
