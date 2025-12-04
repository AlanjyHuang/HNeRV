#!/usr/bin/env python3
"""
Visualize CLIP Embedding Space: Model vs Ground Truth

Creates a publication-quality visualization showing:
1. Ground truth CLIP embeddings in 2D space (t-SNE/UMAP)
2. Model CLIP embeddings in 2D space
3. Statistical comparison panel

This helps demonstrate variance compression and overfitting for presentations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
import argparse
from pathlib import Path
import sys

# Import model components
from model_patch_dual import DualHeadHNeRV, PatchVideoDataSet, CLIPManager
from hnerv_utils import data_split as split_frames


def load_embeddings(checkpoint_path, data_path, crop_list, data_split, num_frames=192):
    """Load both model and ground truth CLIP embeddings."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create args object for model initialization
    class Args:
        pass
    
    args = Args()
    args.embed = 'pe_1.25_80'
    args.ks = '0_3_3'
    args.fc_dim = 96  # Correct fc_dim from checkpoint
    args.fc_hw = '9_16'
    args.num_blks = '1_1'
    args.norm = 'none'
    args.act = 'gelu'
    args.reduce = 1.5
    args.lower_width = 12
    args.dec_strds = [5, 2, 2]
    args.conv_type = ['convnext', 'pshuffel']
    args.clip_dim = 512
    args.out_bias = 'tanh'
    
    # Initialize model
    model = DualHeadHNeRV(args)
    
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    
    # Create args for dataset
    dataset_args = Args()
    dataset_args.data_path = data_path
    dataset_args.crop_list = f'{crop_list[0]}_{crop_list[1]}'
    dataset_args.resize_list = '-1'
    
    # Initialize dataset
    dataset = PatchVideoDataSet(dataset_args)
    
    # Handle train/val split (frame-based)
    num_frames = len(dataset.video)
    split_num_list = [int(x) for x in data_split.split('_')]
    frame_indices = list(range(num_frames))
    train_frame_list, val_frame_list = split_frames(frame_indices, split_num_list, False, 0)
    
    # Create train/val indicator for each patch
    num_patches_per_frame = dataset.num_patches
    train_frames_set = set(train_frame_list)
    
    model_embeddings = []
    gt_embeddings = []
    frame_indices_list = []
    patch_indices_list = []
    is_train_list = []
    
    print(f"Processing {len(dataset)} patches...")
    print(f"Train frames: {len(train_frame_list)}, Val frames: {len(val_frame_list)}")
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            if idx % 100 == 0:
                print(f"Processing patch {idx}/{len(dataset)}")
            
            # Get data
            sample = dataset[idx]
            img = sample['img']
            input_coords = sample['input_coords']
            frame_idx = sample['frame_idx']
            patch_idx = sample['patch_idx']
            is_train = frame_idx in train_frames_set
            
            # Model embedding
            img_batch = img.unsqueeze(0)
            coords_batch = input_coords.unsqueeze(0)
            _, clip_out = model(coords_batch, None)
            model_emb = clip_out.squeeze(0).cpu().numpy()
            
            # Ground truth embedding (from dataset cache)
            gt_emb = sample['clip_embed'].cpu().numpy()
            
            model_embeddings.append(model_emb)
            gt_embeddings.append(gt_emb)
            frame_indices_list.append(frame_idx)
            patch_indices_list.append(patch_idx)
            is_train_list.append(is_train)
    
    model_embeddings = np.array(model_embeddings)
    gt_embeddings = np.array(gt_embeddings)
    frame_indices_list = np.array(frame_indices_list)
    patch_indices_list = np.array(patch_indices_list)
    is_train_list = np.array(is_train_list)
    
    print(f"Loaded {len(model_embeddings)} embeddings")
    print(f"  Train: {is_train_list.sum()}, Val: {(~is_train_list).sum()}")
    
    return {
        'model': model_embeddings,
        'gt': gt_embeddings,
        'frames': frame_indices_list,
        'patches': patch_indices_list,
        'is_train': is_train_list
    }


def compute_statistics(embeddings):
    """Compute variance and distance statistics."""
    # Variance (mean across dimensions)
    variance = np.var(embeddings, axis=0).mean()
    
    # Pairwise distances (sample for efficiency)
    n_samples = min(1000, len(embeddings))
    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    sample = embeddings[indices]
    
    distances = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            dist = np.linalg.norm(sample[i] - sample[j])
            distances.append(dist)
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    return {
        'variance': variance,
        'mean_distance': mean_dist,
        'std_distance': std_dist
    }


def reduce_dimensions(embeddings, method='tsne', random_state=42):
    """Reduce embeddings to 2D using t-SNE or UMAP."""
    print(f"Reducing dimensions using {method.upper()}...")
    
    if method == 'tsne':
        reducer = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=30,
            n_iter=1000,
            verbose=1
        )
    elif method == 'umap':
        reducer = UMAP(
            n_components=2,
            random_state=random_state,
            n_neighbors=15,
            min_dist=0.1,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    embeddings_2d = reducer.fit_transform(embeddings)
    return embeddings_2d


def create_visualization(data, method='tsne', save_path='embedding_space.png'):
    """Create 3-panel publication-quality visualization."""
    
    # Compute statistics
    model_stats = compute_statistics(data['model'])
    gt_stats = compute_statistics(data['gt'])
    
    print("\nStatistics:")
    print(f"Model - Variance: {model_stats['variance']:.6f}, Mean Distance: {model_stats['mean_distance']:.4f}")
    print(f"GT    - Variance: {gt_stats['variance']:.6f}, Mean Distance: {gt_stats['mean_distance']:.4f}")
    print(f"Variance Ratio: {model_stats['variance'] / gt_stats['variance']:.2f}")
    print(f"Distance Ratio: {model_stats['mean_distance'] / gt_stats['mean_distance']:.2f}")
    
    # Reduce dimensions
    model_2d = reduce_dimensions(data['model'], method=method)
    gt_2d = reduce_dimensions(data['gt'], method=method)
    
    # Create figure with 3 panels
    fig = plt.figure(figsize=(18, 5))
    
    # Color by train/val split
    colors = np.where(data['is_train'], 'blue', 'red')
    alpha = 0.6
    
    # Panel 1: Ground Truth embeddings
    ax1 = plt.subplot(131)
    scatter1 = ax1.scatter(
        gt_2d[:, 0], gt_2d[:, 1],
        c=colors, alpha=alpha, s=20, edgecolors='none'
    )
    ax1.set_title('Ground Truth CLIP Embeddings', fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    ax1.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add variance text
    ax1.text(
        0.05, 0.95,
        f'Variance: {gt_stats["variance"]:.6f}\nMean Dist: {gt_stats["mean_distance"]:.4f}',
        transform=ax1.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=10
    )
    
    # Panel 2: Model embeddings
    ax2 = plt.subplot(132)
    scatter2 = ax2.scatter(
        model_2d[:, 0], model_2d[:, 1],
        c=colors, alpha=alpha, s=20, edgecolors='none'
    )
    ax2.set_title('Model CLIP Embeddings', fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    ax2.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add variance text
    ax2.text(
        0.05, 0.95,
        f'Variance: {model_stats["variance"]:.6f}\nMean Dist: {model_stats["mean_distance"]:.4f}',
        transform=ax2.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=10
    )
    
    # Panel 3: Statistical comparison
    ax3 = plt.subplot(133)
    ax3.axis('off')
    
    # Create comparison table
    comparison_text = f"""
EMBEDDING SPACE COMPARISON

Ground Truth Statistics:
  • Variance: {gt_stats['variance']:.6f}
  • Mean Distance: {gt_stats['mean_distance']:.4f}
  • Std Distance: {gt_stats['std_distance']:.4f}

Model Statistics:
  • Variance: {model_stats['variance']:.6f}
  • Mean Distance: {model_stats['mean_distance']:.4f}
  • Std Distance: {model_stats['std_distance']:.4f}

Compression Metrics:
  • Variance Ratio: {model_stats['variance'] / gt_stats['variance']:.2%}
  • Distance Ratio: {model_stats['mean_distance'] / gt_stats['mean_distance']:.2%}

Data Split:
  • Train samples: {data['is_train'].sum()} (blue)
  • Val samples: {(~data['is_train']).sum()} (red)
  • Total samples: {len(data['is_train'])}

Interpretation:
  Lower variance ratio indicates the model
  compresses the embedding space compared
  to ground truth, suggesting overfitting
  and reduced semantic diversity.
"""
    
    ax3.text(
        0.1, 0.95,
        comparison_text,
        transform=ax3.transAxes,
        verticalalignment='top',
        fontsize=11,
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
    )
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=alpha, label='Train'),
        Patch(facecolor='red', alpha=alpha, label='Val')
    ]
    ax3.legend(
        handles=legend_elements,
        loc='lower left',
        frameon=True,
        fontsize=11
    )
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    
    # Also save as PDF for publications
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF version saved to: {pdf_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize CLIP embedding space')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to video data directory')
    parser.add_argument('--crop_list', type=int, nargs=2, default=[2, 4],
                        help='Patch grid size (default: 2 4)')
    parser.add_argument('--data_split', type=str, default='9_10_10',
                        help='Data split ratio (default: 9_10_10)')
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'umap'],
                        help='Dimensionality reduction method (default: tsne)')
    parser.add_argument('--output', type=str, default='embedding_space_comparison.png',
                        help='Output filename (default: embedding_space_comparison.png)')
    parser.add_argument('--num_frames', type=int, default=192,
                        help='Number of frames in video (default: 192)')
    
    args = parser.parse_args()
    
    # Load embeddings
    data = load_embeddings(
        args.checkpoint,
        args.data_path,
        args.crop_list,
        args.data_split,
        args.num_frames
    )
    
    # Create visualization
    create_visualization(data, method=args.method, save_path=args.output)
    
    print("\n✓ Visualization complete!")
    print(f"  Output: {args.output}")
    print(f"  PDF: {args.output.replace('.png', '.pdf')}")


if __name__ == '__main__':
    main()
