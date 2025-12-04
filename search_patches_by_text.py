"""
Text-to-Patch Search Tool
Search for video patches using natural language queries.
Uses CLIP text embeddings to find semantically matching patches.
"""

import torch
import torch.nn.functional as F
import argparse
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import clip

from model_patch_dual import DualHeadHNeRV, PatchVideoDataSet, CLIPManager


def search_patches_by_text(model, dataset, clip_manager, text_query, device, top_k=10):
    """
    Search for patches matching a text query.
    
    Args:
        model: Trained DualHeadHNeRV model
        dataset: PatchVideoDataSet
        clip_manager: CLIPManager for text encoding
        text_query: Text description (e.g., "a silver refrigerator")
        device: torch device
        top_k: Number of top results to return
    
    Returns:
        List of tuples: (similarity_score, frame_idx, patch_idx, rgb_image, clip_embedding)
    """
    print(f"\nSearching for: '{text_query}'")
    print("="*60)
    
    # Encode text query to CLIP embedding
    text_embedding = clip_manager.encode_text(text_query).to(device)
    text_embedding = F.normalize(text_embedding, dim=-1)
    
    print(f"Text embedding shape: {text_embedding.shape}")
    
    # Search through all patches
    model.eval()
    results = []
    
    num_frames = len(dataset) // dataset.num_patches
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Searching patches"):
            data = dataset[idx]
            
            # Get model's CLIP embedding for this patch
            coords = data['input_coords'].unsqueeze(0).to(device)
            rgb_out, clip_out, _, _ = model(coords)
            
            # Normalize CLIP embedding
            clip_out = F.normalize(clip_out, dim=-1)
            
            # Compute similarity with text query
            similarity = (clip_out @ text_embedding.T).item()
            
            # Extract frame and patch info
            frame_idx = idx // dataset.num_patches
            patch_idx = idx % dataset.num_patches
            
            # Compute patch position in grid
            patch_row = patch_idx // dataset.num_patches_w
            patch_col = patch_idx % dataset.num_patches_w
            
            # Convert RGB output to displayable image
            rgb_image = rgb_out.squeeze(0).cpu()  # [3, H, W]
            rgb_image = rgb_image.clamp(0, 1)
            
            results.append({
                'similarity': similarity,
                'frame_idx': frame_idx,
                'patch_idx': patch_idx,
                'patch_row': patch_row,
                'patch_col': patch_col,
                'rgb_image': rgb_image,
                'clip_embedding': clip_out.squeeze(0).cpu()
            })
    
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Return top k results
    return results[:top_k]


def visualize_results(results, text_query, output_dir='search_results'):
    """
    Visualize search results in a grid.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_results = len(results)
    cols = min(5, num_results)
    rows = (num_results + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*4))
    if num_results == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        
        # Display RGB patch
        rgb_np = result['rgb_image'].permute(1, 2, 0).numpy()
        ax.imshow(rgb_np)
        ax.axis('off')
        
        # Add title with info
        title = (f"Rank {idx+1} (sim: {result['similarity']:.4f})\n"
                f"Frame {result['frame_idx']}, "
                f"Patch ({result['patch_row']},{result['patch_col']})")
        ax.set_title(title, fontsize=10)
    
    # Hide unused subplots
    for idx in range(num_results, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"Search Results for: '{text_query}'", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    safe_query = text_query.replace(' ', '_').replace('/', '_')
    output_path = os.path.join(output_dir, f'search_{safe_query}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to: {output_path}")
    plt.close()
    
    return output_path


def save_individual_patches(results, text_query, output_dir='search_results'):
    """
    Save individual patch images with metadata.
    """
    safe_query = text_query.replace(' ', '_').replace('/', '_')
    query_dir = os.path.join(output_dir, safe_query)
    os.makedirs(query_dir, exist_ok=True)
    
    for idx, result in enumerate(results):
        # Save RGB image
        rgb_np = (result['rgb_image'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(rgb_np)
        
        filename = (f"rank{idx+1:02d}_frame{result['frame_idx']:04d}_"
                   f"patch{result['patch_row']}{result['patch_col']}_"
                   f"sim{result['similarity']:.4f}.png")
        img_path = os.path.join(query_dir, filename)
        img.save(img_path)
    
    print(f"Individual patches saved to: {query_dir}")
    return query_dir


def print_results_table(results, text_query):
    """
    Print results in a formatted table.
    """
    print("\n" + "="*80)
    print(f"Top {len(results)} Results for: '{text_query}'")
    print("="*80)
    print(f"{'Rank':<6} {'Similarity':<12} {'Frame':<8} {'Patch':<12} {'Position':<15}")
    print("-"*80)
    
    for idx, result in enumerate(results):
        rank = idx + 1
        sim = result['similarity']
        frame = result['frame_idx']
        patch = result['patch_idx']
        position = f"({result['patch_row']}, {result['patch_col']})"
        
        print(f"{rank:<6} {sim:<12.4f} {frame:<8} {patch:<12} {position:<15}")
    
    print("="*80)


def interactive_search_mode(model, dataset, clip_manager, device, args):
    """
    Interactive mode where user can enter multiple queries.
    """
    print("\n" + "="*80)
    print("Interactive Text-to-Patch Search")
    print("="*80)
    print("Enter text queries to search for patches. Type 'quit' or 'exit' to stop.")
    print("Examples:")
    print("  - 'a silver refrigerator'")
    print("  - 'wooden cabinet'")
    print("  - 'white wall'")
    print("  - 'kitchen countertop'")
    print("="*80)
    
    while True:
        print("\n" + "-"*80)
        text_query = input("Enter search query: ").strip()
        
        if text_query.lower() in ['quit', 'exit', 'q']:
            print("Exiting search mode.")
            break
        
        if not text_query:
            print("Please enter a valid query.")
            continue
        
        # Search for patches
        results = search_patches_by_text(
            model, dataset, clip_manager, text_query, device, top_k=args.top_k
        )
        
        # Print results table
        print_results_table(results, text_query)
        
        # Visualize results
        visualize_results(results, text_query, output_dir=args.output_dir)
        
        # Save individual patches
        if args.save_individual:
            save_individual_patches(results, text_query, output_dir=args.output_dir)


def main():
    parser = argparse.ArgumentParser(description='Search video patches using text queries')
    
    # Model and data parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to video data directory')
    parser.add_argument('--vid', type=str, required=True,
                       help='Video name')
    
    # Model architecture (must match training)
    parser.add_argument('--fc_dim', type=int, default=96)
    parser.add_argument('--fc_hw', type=str, default='9_16')
    parser.add_argument('--ks', type=str, default='0_3_3')
    parser.add_argument('--num_blks', type=str, default='1_1')
    parser.add_argument('--enc_strds', nargs='+', type=int, default=[5, 2, 2])
    parser.add_argument('--dec_strds', nargs='+', type=int, default=[5, 2, 2])
    parser.add_argument('--reduce', type=float, default=1.5)
    parser.add_argument('--lower_width', type=int, default=12)
    parser.add_argument('--conv_type', nargs='+', type=str, default=['convnext', 'pshuffel'])
    parser.add_argument('--norm', type=str, default='none')
    parser.add_argument('--act', type=str, default='gelu')
    parser.add_argument('--embed', type=str, default='pe_1.25_80')
    parser.add_argument('--clip_dim', type=int, default=512)
    parser.add_argument('--out_bias', type=str, default='tanh')
    
    # Data parameters
    parser.add_argument('--data_split', type=str, default='9_10_10')
    parser.add_argument('--crop_list', type=str, default='640_1280')
    parser.add_argument('--resize_list', type=str, default='-1')
    
    # Search parameters
    parser.add_argument('--query', type=str, default=None,
                       help='Text query (if not provided, enters interactive mode)')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of top results to return')
    parser.add_argument('--output_dir', type=str, default='search_results',
                       help='Directory to save results')
    parser.add_argument('--save_individual', action='store_true',
                       help='Save individual patch images')
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive search mode')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize CLIP
    print("\nInitializing CLIP model...")
    clip_manager = CLIPManager(device=device)
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_path}...")
    dataset = PatchVideoDataSet(args)
    num_frames = len(dataset) // dataset.num_patches
    print(f"Dataset: {num_frames} frames, {dataset.num_patches} patches/frame, "
          f"{len(dataset)} total patches")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = DualHeadHNeRV(args).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully")
    
    # Run search
    if args.interactive or args.query is None:
        # Interactive mode
        interactive_search_mode(model, dataset, clip_manager, device, args)
    else:
        # Single query mode
        results = search_patches_by_text(
            model, dataset, clip_manager, args.query, device, top_k=args.top_k
        )
        
        # Print results
        print_results_table(results, args.query)
        
        # Visualize results
        visualize_results(results, args.query, output_dir=args.output_dir)
        
        # Save individual patches
        if args.save_individual:
            save_individual_patches(results, args.query, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
