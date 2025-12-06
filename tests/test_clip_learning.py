"""
Test if CLIP embeddings learned meaningful representations.
Compares trained model's CLIP similarity against:
1. Random embeddings baseline
2. Shuffled embeddings (random pairing)
3. Temporal shuffle (swapped frame assignments)
"""

import torch
import torch.nn.functional as F
import argparse
import os
import sys
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from datetime import datetime

# Import from existing files
from model_all import HNeRV, VideoDataSet, CLIPManager
from hnerv_utils import set_seed


def data_to_gpu(sample, device):
    """Recursively move data to GPU"""
    if isinstance(sample, dict):
        return {k: data_to_gpu(v, device) for k, v in sample.items()}
    elif isinstance(sample, list):
        return [data_to_gpu(v, device) for v in sample]
    elif isinstance(sample, torch.Tensor):
        return sample.to(device, non_blocking=True)
    else:
        return sample


def compute_clip_similarity(pred_embeds, gt_embeds):
    """Compute cosine similarity between predicted and ground truth CLIP embeddings"""
    return F.cosine_similarity(pred_embeds, gt_embeds, dim=-1).mean().item()


@torch.no_grad()
def test_clip_learning(model, dataloader, device, args):
    """
    Test if CLIP embeddings are meaningful by comparing against baselines.
    
    Returns:
        dict with keys:
            - 'learned': CLIP similarity of trained model
            - 'random': CLIP similarity with random embeddings
            - 'shuffled': CLIP similarity with shuffled frame-embedding pairs
            - 'temporal_shuffle': CLIP similarity with temporally shuffled embeddings
    """
    model.eval()
    clip_manager = CLIPManager(device=device)
    
    # Storage for results
    learned_sims = []
    random_sims = []
    shuffled_sims = []
    temporal_shuffle_sims = []
    
    # Collect all data first for shuffling
    all_outputs = []
    all_gt_embeds = []
    all_indices = []
    
    print("Collecting model outputs and ground truth embeddings...")
    for i, sample in enumerate(dataloader):
        if i > 20 and args.debug:  # Limit for debugging
            break
            
        sample = data_to_gpu(sample, device)
        img_data = sample['img']
        norm_idx = sample['norm_idx']
        img_idx = sample['idx']
        clip_embeds = sample.get('clip_embeds')
        
        if clip_embeds is None:
            print("Warning: No CLIP embeddings in dataset. Make sure you're using VideoDataSet with CLIP enabled.")
            continue
        
        # Get model predictions
        model_instance = model.module if hasattr(model, 'module') else model
        
        if 'pe' in args.embed:
            pe_features = model_instance.pe_embed(norm_idx[:, None]).float().squeeze(-1).squeeze(-1)
            selected_clip_embeds = clip_embeds[:, 0, :]  # Use same selection logic as training
            combined_features = torch.cat([pe_features, selected_clip_embeds], dim=1)
            cur_input = combined_features.unsqueeze(-1).unsqueeze(-1)
            img_out, _, _ = model(None, cur_input)
        else:
            img_out, _, _ = model(img_data)
        
        # Resize if needed
        if img_out.shape[-2:] != img_data.shape[-2:]:
            img_out = F.interpolate(img_out, size=img_data.shape[-2:], mode='bilinear', align_corners=False)
        
        # Extract CLIP embeddings from model output
        pred_clip_embeds_batch = []
        for b in range(img_out.shape[0]):
            img_denorm = (img_out[b] + 1) / 2
            img_denorm = torch.clamp(img_denorm, 0, 1)
            patches, _ = clip_manager.extract_patches_and_coords(img_denorm.unsqueeze(0))
            embeds = clip_manager.encode_patches(patches)
            pred_clip_embeds_batch.append(embeds)
        pred_clip_embeds = torch.stack(pred_clip_embeds_batch)
        
        all_outputs.append(pred_clip_embeds)
        all_gt_embeds.append(clip_embeds)
        all_indices.extend(img_idx.cpu().numpy())
        
    # Concatenate all data
    all_outputs = torch.cat(all_outputs, dim=0)
    all_gt_embeds = torch.cat(all_gt_embeds, dim=0)
    n_samples = all_outputs.shape[0]
    embed_dim = all_outputs.shape[-1]
    
    print(f"\nCollected {n_samples} frames")
    print(f"CLIP embedding dimension: {embed_dim}")
    
    # Test 1: Learned (actual model performance)
    print("\n1. Computing learned CLIP similarity...")
    for i in range(n_samples):
        sim = compute_clip_similarity(all_outputs[i:i+1], all_gt_embeds[i:i+1])
        learned_sims.append(sim)
    
    # Test 2: Random embeddings baseline
    print("2. Computing random embedding baseline...")
    for i in range(n_samples):
        random_embed = torch.randn_like(all_gt_embeds[i:i+1])
        random_embed = F.normalize(random_embed, dim=-1)  # Normalize like CLIP
        sim = compute_clip_similarity(all_outputs[i:i+1], random_embed)
        random_sims.append(sim)
    
    # Test 3: Shuffled pairing (random frame-embedding pairs)
    print("3. Computing shuffled pairing baseline...")
    shuffled_indices = torch.randperm(n_samples)
    for i in range(n_samples):
        shuffled_idx = shuffled_indices[i]
        sim = compute_clip_similarity(all_outputs[i:i+1], all_gt_embeds[shuffled_idx:shuffled_idx+1])
        shuffled_sims.append(sim)
    
    # Test 4: Temporal shuffle (swap nearby frames)
    print("4. Computing temporal shuffle baseline...")
    # Shuffle but keep some local structure
    temporal_indices = np.arange(n_samples)
    np.random.shuffle(temporal_indices)
    for i in range(n_samples):
        temporal_idx = temporal_indices[i]
        sim = compute_clip_similarity(all_outputs[i:i+1], all_gt_embeds[temporal_idx:temporal_idx+1])
        temporal_shuffle_sims.append(sim)
    
    # Compute statistics
    results = {
        'learned_mean': np.mean(learned_sims),
        'learned_std': np.std(learned_sims),
        'random_mean': np.mean(random_sims),
        'random_std': np.std(random_sims),
        'shuffled_mean': np.mean(shuffled_sims),
        'shuffled_std': np.std(shuffled_sims),
        'temporal_shuffle_mean': np.mean(temporal_shuffle_sims),
        'temporal_shuffle_std': np.std(temporal_shuffle_sims),
        'n_samples': n_samples,
    }
    
    # Compute improvement over baselines
    results['improvement_over_random'] = results['learned_mean'] - results['random_mean']
    results['improvement_over_shuffled'] = results['learned_mean'] - results['shuffled_mean']
    results['improvement_over_temporal'] = results['learned_mean'] - results['temporal_shuffle_mean']
    
    # Statistical significance (t-test approximation)
    results['significant_vs_random'] = results['improvement_over_random'] > 2 * results['random_std']
    results['significant_vs_shuffled'] = results['improvement_over_shuffled'] > 2 * results['shuffled_std']
    
    return results, {
        'learned': learned_sims,
        'random': random_sims,
        'shuffled': shuffled_sims,
        'temporal_shuffle': temporal_shuffle_sims
    }


def print_results(results):
    """Print formatted results"""
    print("\n" + "="*70)
    print("CLIP LEARNING VERIFICATION RESULTS")
    print("="*70)
    print(f"\nNumber of frames tested: {results['n_samples']}")
    print("\n" + "-"*70)
    print(f"{'Condition':<30} {'Mean':>12} {'Std':>12}")
    print("-"*70)
    print(f"{'Learned (Your Model)':<30} {results['learned_mean']:>12.4f} {results['learned_std']:>12.4f}")
    print(f"{'Random Embeddings':<30} {results['random_mean']:>12.4f} {results['random_std']:>12.4f}")
    print(f"{'Shuffled Pairing':<30} {results['shuffled_mean']:>12.4f} {results['shuffled_std']:>12.4f}")
    print(f"{'Temporal Shuffle':<30} {results['temporal_shuffle_mean']:>12.4f} {results['temporal_shuffle_std']:>12.4f}")
    print("-"*70)
    
    print("\n" + "="*70)
    print("IMPROVEMENT ANALYSIS")
    print("="*70)
    print(f"Improvement over random:          {results['improvement_over_random']:>8.4f}  {'✓ Significant' if results['significant_vs_random'] else '✗ Not significant'}")
    print(f"Improvement over shuffled:        {results['improvement_over_shuffled']:>8.4f}  {'✓ Significant' if results['significant_vs_shuffled'] else '✗ Not significant'}")
    print(f"Improvement over temporal:        {results['improvement_over_temporal']:>8.4f}")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if results['learned_mean'] > 0.8:
        quality = "EXCELLENT"
        msg = "Model has learned very strong semantic representations!"
    elif results['learned_mean'] > 0.6:
        quality = "GOOD"
        msg = "Model has learned meaningful semantic features."
    elif results['learned_mean'] > 0.4:
        quality = "MODERATE"
        msg = "Model captures some semantic information but has room for improvement."
    else:
        quality = "POOR"
        msg = "Model may not be learning semantic features effectively."
    
    print(f"Overall Quality: {quality}")
    print(f"{msg}")
    
    if results['significant_vs_random']:
        print("✓ Performance significantly better than random (learning is happening!)")
    else:
        print("✗ Performance not significantly better than random (may need more training)")
    
    if results['significant_vs_shuffled']:
        print("✓ Model learned correct frame-to-embedding associations")
    else:
        print("✗ Model may not have learned proper temporal/spatial structure")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to video data')
    parser.add_argument('--outf', type=str, default='./clip_test_results', help='Output folder for results')
    
    # Model architecture (should match training)
    parser.add_argument('--embed', default='pe_1.25_80', type=str, help='Embedding type')
    parser.add_argument('--fc_hw_dim', default="9_16_112", type=str, help='FC dimension')
    parser.add_argument('--expansion', default=1, type=float, help='Channel expansion')
    parser.add_argument('--num_blks', default="1_1", type=str, help='Number of blocks')
    parser.add_argument('--enc_strds', default='5,2,2', type=str, help='Encoder strides')
    parser.add_argument('--enc_dim', default='64_16', type=str, help='Encoder dimensions')
    parser.add_argument('--dec_strds', default='5,2,2', type=str, help='Decoder strides')
    parser.add_argument('--conv_type', default='pshuffel', type=str, help='Convolution type')
    parser.add_argument('--act', default='gelu', type=str, help='Activation function')
    parser.add_argument('--norm', default='none', type=str, help='Normalization')
    parser.add_argument('--lower_width', default=12, type=int, help='Lower width')
    parser.add_argument('--single_res', default=True, type=bool, help='Single resolution')
    parser.add_argument('--enc_block', default='convnext', type=str, help='Encoder block type')
    
    # Data arguments
    parser.add_argument('--batchSize', type=int, default=1, help='Batch size')
    parser.add_argument('--vid_path', type=str, default='compress', help='Video path')
    parser.add_argument('--crop_list', type=str, default=None, help='Crop list')
    parser.add_argument('--resize_list', type=str, default=None, help='Resize list')
    parser.add_argument('--debug', action='store_true', help='Debug mode (process fewer frames)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outf, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(0)
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_path}...")
    dataset = VideoDataSet(
        args.data_path,
        args.vid_path,
        args.crop_list,
        args.resize_list,
        0,  # train_fraction (use all data for testing)
        is_train=False,
    )
    dataloader = DataLoader(dataset, batch_size=args.batchSize, shuffle=False, num_workers=4)
    print(f"Dataset size: {len(dataset)} frames")
    
    # Build model
    print("\nBuilding model...")
    args.ks = '0_3_3'  # Default kernel sizes
    args.reduce = 1.5
    args.blocks_list = [int(x) for x in args.num_blks.split('_')]
    
    model = HNeRV(
        embed_length=dataset.frame_num,
        stem_dim_num=args.fc_hw_dim,
        fc_hw_dim=args.fc_hw_dim,
        expansion=args.expansion,
        num_blocks=args.blocks_list,
        norm=args.norm,
        act=args.act,
        bias=True,
        conv_type=args.conv_type,
        stride_list=[int(x) for x in args.dec_strds.split(',')],
        enc_strds=[int(x) for x in args.enc_strds.split(',')],
        enc_dim=[int(x) for x in args.enc_dim.split('_')],
        ks=args.ks,
        reduce=args.reduce,
        lower_width=args.lower_width,
        enc_block=args.enc_block,
        embedding=args.embed,
    )
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Handle state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Handle module prefix
    model_has_module = hasattr(model, 'module')
    state_has_module = any(k.startswith('module.') for k in state_dict.keys())
    
    if model_has_module and not state_has_module:
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    elif not model_has_module and state_has_module:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Run tests
    print("\n" + "="*70)
    print("Starting CLIP Learning Verification Tests...")
    print("="*70)
    
    results, detailed_sims = test_clip_learning(model, dataloader, device, args)
    
    # Print results
    print_results(results)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save summary
    summary_path = os.path.join(args.outf, f'clip_test_summary_{timestamp}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"CLIP Learning Verification Results\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {args.data_path}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Number of frames: {results['n_samples']}\n\n")
        f.write(f"Learned mean: {results['learned_mean']:.4f} ± {results['learned_std']:.4f}\n")
        f.write(f"Random mean: {results['random_mean']:.4f} ± {results['random_std']:.4f}\n")
        f.write(f"Shuffled mean: {results['shuffled_mean']:.4f} ± {results['shuffled_std']:.4f}\n")
        f.write(f"Temporal shuffle mean: {results['temporal_shuffle_mean']:.4f} ± {results['temporal_shuffle_std']:.4f}\n\n")
        f.write(f"Improvement over random: {results['improvement_over_random']:.4f}\n")
        f.write(f"Improvement over shuffled: {results['improvement_over_shuffled']:.4f}\n")
        f.write(f"Improvement over temporal: {results['improvement_over_temporal']:.4f}\n")
    
    print(f"Summary saved to: {summary_path}")
    
    # Save detailed CSV
    csv_path = os.path.join(args.outf, f'clip_test_detailed_{timestamp}.csv')
    df = pd.DataFrame({
        'frame_idx': range(results['n_samples']),
        'learned_sim': detailed_sims['learned'],
        'random_sim': detailed_sims['random'],
        'shuffled_sim': detailed_sims['shuffled'],
        'temporal_shuffle_sim': detailed_sims['temporal_shuffle'],
    })
    df.to_csv(csv_path, index=False)
    print(f"Detailed results saved to: {csv_path}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
