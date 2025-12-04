"""
Verification script to test if generated CLIP embeddings are semantically meaningful
and not just random interpolations.
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
from tqdm import tqdm
import json

# Import your model and data utilities
from model_patch_dual import DualHeadHNeRV, PatchVideoDataSet, CLIPManager


def compute_embedding_statistics(embeddings):
    """Compute statistics to detect embedding collapse."""
    embeddings_np = embeddings.cpu().numpy()
    
    # Compute variance across embeddings
    variance = np.var(embeddings_np, axis=0).mean()
    
    # Compute pairwise distances
    n = len(embeddings)
    distances = []
    for i in range(min(n, 100)):  # Sample for efficiency
        for j in range(i+1, min(n, 100)):
            dist = np.linalg.norm(embeddings_np[i] - embeddings_np[j])
            distances.append(dist)
    
    mean_distance = np.mean(distances) if distances else 0
    std_distance = np.std(distances) if distances else 0
    
    return {
        'variance': float(variance),
        'mean_pairwise_distance': float(mean_distance),
        'std_pairwise_distance': float(std_distance),
        'min_distance': float(np.min(distances)) if distances else 0,
        'max_distance': float(np.max(distances)) if distances else 0
    }


def test_ground_truth_similarity(model, dataset, clip_manager, device, args, max_samples=None):
    """
    Test 1: Compare generated CLIP embeddings with ground truth CLIP embeddings.
    High similarity (>0.9) indicates embeddings are meaningful.
    """
    print("\n" + "="*60)
    print("Test 1: Ground Truth CLIP Similarity")
    print("="*60)
    
    model.eval()
    similarities = []
    train_similarities = []
    val_similarities = []
    
    # Determine data split
    total_frames = len(dataset) // dataset.num_patches
    if args.data_split:
        split_parts = [int(x) for x in args.data_split.split('_')]
        valid_train = split_parts[0]
        total_data_length = split_parts[2]
    else:
        valid_train = 9
        total_data_length = 10
    
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Computing similarities"):
            data = dataset[idx]
            
            # Get model's generated CLIP embedding
            coords = data['input_coords'].unsqueeze(0).to(device)
            _, clip_out, _, _ = model(coords)
            generated_clip = F.normalize(clip_out.squeeze(0), dim=0)
            
            # Get ground truth CLIP embedding
            ground_truth_clip = F.normalize(data['clip_embed'].to(device), dim=0)
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(generated_clip.unsqueeze(0), 
                                            ground_truth_clip.unsqueeze(0)).item()
            similarities.append(similarity)
            
            # Track train vs val
            frame_idx = data['frame_idx']
            position_in_group = frame_idx % total_data_length
            is_train = position_in_group < valid_train
            
            if is_train:
                train_similarities.append(similarity)
            else:
                val_similarities.append(similarity)
    
    results = {
        'overall_mean': float(np.mean(similarities)),
        'overall_std': float(np.std(similarities)),
        'overall_min': float(np.min(similarities)),
        'overall_max': float(np.max(similarities)),
        'train_mean': float(np.mean(train_similarities)) if train_similarities else 0,
        'train_std': float(np.std(train_similarities)) if train_similarities else 0,
        'val_mean': float(np.mean(val_similarities)) if val_similarities else 0,
        'val_std': float(np.std(val_similarities)) if val_similarities else 0,
        'num_samples': num_samples,
        'num_train': len(train_similarities),
        'num_val': len(val_similarities)
    }
    
    print(f"\nOverall Similarity: {results['overall_mean']:.4f} ± {results['overall_std']:.4f}")
    print(f"  Min: {results['overall_min']:.4f}, Max: {results['overall_max']:.4f}")
    print(f"\nTrain Similarity: {results['train_mean']:.4f} ± {results['train_std']:.4f} (n={results['num_train']})")
    print(f"Val Similarity:   {results['val_mean']:.4f} ± {results['val_std']:.4f} (n={results['num_val']})")
    
    # Interpretation
    print("\n" + "-"*60)
    if results['overall_mean'] > 0.9:
        print("✓ EXCELLENT: Embeddings are highly aligned with ground truth")
    elif results['overall_mean'] > 0.7:
        print("✓ GOOD: Embeddings capture semantic content")
    elif results['overall_mean'] > 0.5:
        print("⚠ MODERATE: Embeddings have some semantic content")
    else:
        print("✗ POOR: Embeddings may be random or poorly learned")
    
    return results, similarities


def test_embedding_diversity(model, dataset, device, max_samples=None):
    """
    Test 2: Check if embeddings have sufficient diversity.
    Low variance or similar embeddings suggest collapse/interpolation.
    """
    print("\n" + "="*60)
    print("Test 2: Embedding Diversity & Collapse Detection")
    print("="*60)
    
    model.eval()
    embeddings = []
    
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Collecting embeddings"):
            data = dataset[idx]
            coords = data['input_coords'].unsqueeze(0).to(device)
            _, clip_out, _, _ = model(coords)
            embeddings.append(clip_out.squeeze(0))
    
    embeddings = torch.stack(embeddings)
    stats = compute_embedding_statistics(embeddings)
    
    print(f"\nEmbedding Statistics:")
    print(f"  Variance: {stats['variance']:.6f}")
    print(f"  Mean pairwise distance: {stats['mean_pairwise_distance']:.4f}")
    print(f"  Std pairwise distance: {stats['std_pairwise_distance']:.4f}")
    print(f"  Distance range: [{stats['min_distance']:.4f}, {stats['max_distance']:.4f}]")
    
    # Interpretation
    print("\n" + "-"*60)
    if stats['variance'] < 0.001:
        print("✗ COLLAPSED: Embeddings have very low variance (likely collapsed)")
    elif stats['variance'] < 0.01:
        print("⚠ LOW DIVERSITY: Embeddings may be too similar")
    else:
        print("✓ GOOD DIVERSITY: Embeddings show healthy variation")
    
    if stats['mean_pairwise_distance'] < 0.1:
        print("✗ COLLAPSED: Embeddings are too similar to each other")
    elif stats['mean_pairwise_distance'] < 0.5:
        print("⚠ MODERATE: Some diversity but could be better")
    else:
        print("✓ DIVERSE: Embeddings are well-separated")
    
    return stats, embeddings


def test_temporal_consistency(model, dataset, device):
    """
    Test 3: Check temporal consistency.
    Embeddings should change smoothly but meaningfully across frames.
    """
    print("\n" + "="*60)
    print("Test 3: Temporal Consistency")
    print("="*60)
    
    model.eval()
    
    # Get embeddings for consecutive frames (first patch of each frame)
    num_frames = len(dataset) // dataset.num_patches
    frame_embeddings = []
    
    with torch.no_grad():
        for frame_idx in tqdm(range(min(num_frames, 100)), desc="Processing frames"):
            # Get first patch of each frame
            patch_idx = frame_idx * dataset.num_patches
            data = dataset[patch_idx]
            
            coords = data['input_coords'].unsqueeze(0).to(device)
            _, clip_out, _, _ = model(coords)
            frame_embeddings.append(clip_out.squeeze(0))
    
    frame_embeddings = torch.stack(frame_embeddings)
    
    # Compute frame-to-frame differences
    frame_diffs = []
    for i in range(len(frame_embeddings) - 1):
        diff = torch.norm(frame_embeddings[i+1] - frame_embeddings[i]).item()
        frame_diffs.append(diff)
    
    results = {
        'mean_frame_diff': float(np.mean(frame_diffs)),
        'std_frame_diff': float(np.std(frame_diffs)),
        'max_frame_diff': float(np.max(frame_diffs)),
        'min_frame_diff': float(np.min(frame_diffs))
    }
    
    print(f"\nFrame-to-frame change:")
    print(f"  Mean: {results['mean_frame_diff']:.4f}")
    print(f"  Std: {results['std_frame_diff']:.4f}")
    print(f"  Range: [{results['min_frame_diff']:.4f}, {results['max_frame_diff']:.4f}]")
    
    # Interpretation
    print("\n" + "-"*60)
    if results['std_frame_diff'] < 0.01:
        print("✗ TOO SMOOTH: Embeddings barely change (likely interpolation)")
    elif results['mean_frame_diff'] < 0.05:
        print("⚠ VERY SMOOTH: Changes are subtle (could be over-smoothed)")
    else:
        print("✓ DYNAMIC: Embeddings change meaningfully across frames")
    
    return results


def test_train_val_difference(model, dataset, device, args, max_samples=None):
    """
    Test 4: Compare train vs val embedding quality.
    Large difference suggests overfitting to training data.
    """
    print("\n" + "="*60)
    print("Test 4: Train vs Val Embedding Quality")
    print("="*60)
    
    model.eval()
    
    # Determine data split
    if args.data_split:
        split_parts = [int(x) for x in args.data_split.split('_')]
        valid_train = split_parts[0]
        total_data_length = split_parts[2]
    else:
        valid_train = 9
        total_data_length = 10
    
    train_embeddings = []
    val_embeddings = []
    
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Separating train/val"):
            data = dataset[idx]
            frame_idx = data['frame_idx']
            position_in_group = frame_idx % total_data_length
            is_train = position_in_group < valid_train
            
            coords = data['input_coords'].unsqueeze(0).to(device)
            _, clip_out, _, _ = model(coords)
            
            if is_train:
                train_embeddings.append(clip_out.squeeze(0))
            else:
                val_embeddings.append(clip_out.squeeze(0))
    
    if not val_embeddings:
        print("⚠ No validation samples found!")
        return {}
    
    train_embeddings = torch.stack(train_embeddings)
    val_embeddings = torch.stack(val_embeddings)
    
    train_stats = compute_embedding_statistics(train_embeddings)
    val_stats = compute_embedding_statistics(val_embeddings)
    
    results = {
        'train_variance': train_stats['variance'],
        'val_variance': val_stats['variance'],
        'train_mean_dist': train_stats['mean_pairwise_distance'],
        'val_mean_dist': val_stats['mean_pairwise_distance'],
        'num_train': len(train_embeddings),
        'num_val': len(val_embeddings)
    }
    
    print(f"\nTrain embeddings (n={results['num_train']}):")
    print(f"  Variance: {results['train_variance']:.6f}")
    print(f"  Mean distance: {results['train_mean_dist']:.4f}")
    
    print(f"\nVal embeddings (n={results['num_val']}):")
    print(f"  Variance: {results['val_variance']:.6f}")
    print(f"  Mean distance: {results['val_mean_dist']:.4f}")
    
    # Interpretation
    print("\n" + "-"*60)
    variance_ratio = results['val_variance'] / results['train_variance'] if results['train_variance'] > 0 else 0
    if variance_ratio > 0.8:
        print("✓ CONSISTENT: Train and val embeddings have similar diversity")
    elif variance_ratio > 0.5:
        print("⚠ MODERATE: Some difference in train/val embedding quality")
    else:
        print("✗ OVERFITTING: Val embeddings much less diverse than train")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Verify CLIP embedding quality')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to video data directory')
    parser.add_argument('--vid', type=str, required=True,
                       help='Video name')
    
    # Model architecture
    parser.add_argument('--fc_dim', type=int, default=96)
    parser.add_argument('--fc_hw', type=str, default='9_16', help='FC output size (h,w)')
    parser.add_argument('--ks', type=str, default='0_3_3', help='kernel sizes')
    parser.add_argument('--num_blks', type=str, default='1_1', help='number of blocks')
    parser.add_argument('--enc_strds', nargs='+', type=int, default=[5, 2, 2])
    parser.add_argument('--dec_strds', nargs='+', type=int, default=[5, 2, 2])
    parser.add_argument('--reduce', type=float, default=1.5)
    parser.add_argument('--lower_width', type=int, default=12)
    parser.add_argument('--conv_type', nargs='+', type=str, default=['convnext', 'pshuffel'])
    parser.add_argument('--norm', type=str, default='none')
    parser.add_argument('--act', type=str, default='gelu')
    parser.add_argument('--embed', type=str, default='pe_1.25_80')
    parser.add_argument('--clip_dim', type=int, default=512, help='CLIP embedding dimension')
    parser.add_argument('--out_bias', type=str, default='tanh', help='Output bias')
    
    # Data parameters
    parser.add_argument('--data_split', type=str, default='6_6_10',
                       help='Data split pattern (train_train_total)')
    parser.add_argument('--crop_list', type=str, default='640_1280',
                       help='Crop size as string (e.g., 640_1280)')
    parser.add_argument('--resize_list', type=str, default='-1',
                       help='Resize parameters')
    parser.add_argument('--patch_grid', nargs='+', type=int, default=[2, 4])
    
    # Test parameters
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to test (None = all)')
    parser.add_argument('--output', type=str, default='clip_verification_results.json',
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize CLIP
    print("\nInitializing CLIP model...")
    clip_manager = CLIPManager(device=device)
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_path}...")
    dataset = PatchVideoDataSet(args)
    print(f"Dataset size: {len(dataset)} patches")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = DualHeadHNeRV(args).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully")
    
    # Run all tests
    all_results = {}
    
    # Test 1: Ground truth similarity
    test1_results, similarities = test_ground_truth_similarity(
        model, dataset, clip_manager, device, args, args.max_samples
    )
    all_results['ground_truth_similarity'] = test1_results
    
    # Test 2: Embedding diversity
    test2_results, embeddings = test_embedding_diversity(
        model, dataset, device, args.max_samples
    )
    all_results['embedding_diversity'] = test2_results
    
    # Test 3: Temporal consistency
    test3_results = test_temporal_consistency(model, dataset, device)
    all_results['temporal_consistency'] = test3_results
    
    # Test 4: Train vs val
    test4_results = test_train_val_difference(model, dataset, device, args, args.max_samples)
    all_results['train_val_comparison'] = test4_results
    
    # Overall assessment
    print("\n" + "="*60)
    print("OVERALL ASSESSMENT")
    print("="*60)
    
    overall_score = 0
    max_score = 0
    
    # Scoring criteria
    if test1_results['overall_mean'] > 0.9:
        overall_score += 3
        print("✓ Ground truth similarity: EXCELLENT")
    elif test1_results['overall_mean'] > 0.7:
        overall_score += 2
        print("✓ Ground truth similarity: GOOD")
    elif test1_results['overall_mean'] > 0.5:
        overall_score += 1
        print("⚠ Ground truth similarity: MODERATE")
    else:
        print("✗ Ground truth similarity: POOR")
    max_score += 3
    
    if test2_results['variance'] > 0.01 and test2_results['mean_pairwise_distance'] > 0.5:
        overall_score += 2
        print("✓ Embedding diversity: GOOD")
    elif test2_results['variance'] > 0.001:
        overall_score += 1
        print("⚠ Embedding diversity: MODERATE")
    else:
        print("✗ Embedding diversity: POOR (collapsed)")
    max_score += 2
    
    if test3_results['std_frame_diff'] > 0.01:
        overall_score += 1
        print("✓ Temporal dynamics: GOOD")
    else:
        print("⚠ Temporal dynamics: TOO SMOOTH")
    max_score += 1
    
    if test4_results and 'val_variance' in test4_results:
        variance_ratio = test4_results['val_variance'] / test4_results['train_variance'] if test4_results['train_variance'] > 0 else 0
        if variance_ratio > 0.8:
            overall_score += 2
            print("✓ Generalization: EXCELLENT")
        elif variance_ratio > 0.5:
            overall_score += 1
            print("⚠ Generalization: MODERATE")
        else:
            print("✗ Generalization: POOR (overfitting)")
        max_score += 2
    
    print("\n" + "="*60)
    print(f"FINAL SCORE: {overall_score}/{max_score}")
    print("="*60)
    
    if overall_score >= max_score * 0.8:
        print("\n✓✓✓ EMBEDDINGS ARE SEMANTICALLY MEANINGFUL ✓✓✓")
        print("Your model is learning real features, not just interpolating!")
    elif overall_score >= max_score * 0.6:
        print("\n✓ Embeddings are mostly meaningful with some concerns")
    else:
        print("\n✗ WARNING: Embeddings may not be learning properly")
        print("Consider: more regularization, different loss weights, or longer training")
    
    # Save results
    all_results['overall_score'] = {
        'score': overall_score,
        'max_score': max_score,
        'percentage': (overall_score / max_score * 100) if max_score > 0 else 0
    }
    
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
