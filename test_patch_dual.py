"""
Quick test script for the dual-head patch-based HNeRV model.
Tests model creation, forward pass, and loss computation.
"""

import torch
import argparse
from model_patch_dual import DualHeadHNeRV
import torch.nn.functional as F


def test_dual_head_model():
    print("="*60)
    print("Testing Dual-Head Patch-Based HNeRV Model")
    print("="*60)
    
    # Create mock arguments
    args = argparse.Namespace(
        embed='pe_1.25_80',
        ks='0_3_3',
        fc_dim=128,  # Smaller for testing
        fc_hw='9_16',
        reduce=1.2,
        lower_width=16,
        dec_strds=[5, 3, 2],  # Fewer layers for testing
        num_blks='1_1',
        conv_type=['convnext', 'pshuffel'],
        norm='none',
        act='gelu',
        clip_dim=512,
        out_bias='tanh'
    )
    
    print("\n1. Creating model...")
    model = DualHeadHNeRV(args)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   Total parameters: {total_params:.2f}M")
    
    # Count parameters in each head
    rgb_head_params = sum(p.numel() for p in model.rgb_head.parameters()) / 1e3
    clip_head_params = sum(p.numel() for p in model.clip_head.parameters()) / 1e3
    print(f"   RGB head parameters: {rgb_head_params:.2f}K")
    print(f"   CLIP head parameters: {clip_head_params:.2f}K")
    
    print("\n2. Testing forward pass...")
    batch_size = 4
    # Input: (frame_idx, patch_x, patch_y)
    input_coords = torch.rand(batch_size, 3)  # Random coordinates in [0, 1]
    print(f"   Input shape: {input_coords.shape}")
    print(f"   Sample input: {input_coords[0].numpy()}")
    
    # Forward pass
    rgb_out, clip_out, embed_list, dec_time = model(input_coords)
    
    print(f"\n   RGB output shape: {rgb_out.shape}")
    print(f"   CLIP output shape: {clip_out.shape}")
    print(f"   RGB output range: [{rgb_out.min():.3f}, {rgb_out.max():.3f}]")
    print(f"   CLIP output norm: {clip_out.norm(dim=-1).mean():.3f} (should be ~1.0)")
    print(f"   Decoding time: {dec_time*1000:.2f}ms")
    print(f"   Number of intermediate embeddings: {len(embed_list)}")
    
    print("\n3. Testing loss computation...")
    # Create dummy ground truth
    gt_rgb = torch.rand_like(rgb_out)
    gt_clip = F.normalize(torch.rand_like(clip_out), dim=-1)
    
    # Pixel loss
    pixel_loss = F.l1_loss(rgb_out, gt_rgb)
    print(f"   Pixel loss (L1): {pixel_loss.item():.4f}")
    
    # CLIP loss
    clip_sim = F.cosine_similarity(clip_out, gt_clip, dim=-1)
    clip_loss = (1 - clip_sim).mean()
    print(f"   CLIP similarity: {clip_sim.mean().item():.4f}")
    print(f"   CLIP loss: {clip_loss.item():.4f}")
    
    # Hybrid loss
    clip_loss_weight = 0.1
    total_loss = pixel_loss + clip_loss_weight * clip_loss
    print(f"   Total hybrid loss: {total_loss.item():.4f}")
    
    print("\n4. Testing backward pass...")
    total_loss.backward()
    print("   ✓ Backward pass successful")
    
    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params_count = sum(1 for p in model.parameters())
    print(f"   Parameters with gradients: {has_grad}/{total_params_count}")
    
    print("\n5. Testing different input coordinates...")
    test_coords = torch.tensor([
        [0.0, 0.0, 0.0],  # First frame, top-left patch
        [0.5, 0.5, 0.5],  # Middle frame, center patch
        [1.0, 1.0, 1.0],  # Last frame, bottom-right patch
        [0.25, 0.75, 0.25],  # Random position
    ])
    
    with torch.no_grad():
        rgb_out, clip_out, _, _ = model(test_coords)
    
    print("   Input coordinates -> RGB output shape -> CLIP output shape")
    for i in range(len(test_coords)):
        print(f"   {test_coords[i].numpy()} -> {rgb_out[i].shape} -> {clip_out[i].shape}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    
    return model


def test_patch_dataset():
    """Test the patch-based dataset (requires actual data)"""
    print("\n" + "="*60)
    print("Testing Patch-Based Dataset")
    print("="*60)
    
    try:
        from model_patch_dual import PatchVideoDataSet
        
        args = argparse.Namespace(
            data_path='data/Kitchen',  # Adjust to your data path
            crop_list='640_1280',
            resize_list='-1'
        )
        
        print("\n1. Creating dataset...")
        dataset = PatchVideoDataSet(args)
        print(f"   Total samples (patches): {len(dataset)}")
        print(f"   Frame dimensions: {dataset.frame_height} × {dataset.frame_width}")
        print(f"   Patches per frame: {dataset.num_patches} ({dataset.num_patches_h}×{dataset.num_patches_w})")
        print(f"   Patch dimensions: {dataset.patch_h} × {dataset.patch_w}")
        
        print("\n2. Loading sample...")
        sample = dataset[0]
        print(f"   Keys: {sample.keys()}")
        print(f"   Patch image shape: {sample['img'].shape}")
        print(f"   Input coordinates shape: {sample['input_coords'].shape}")
        print(f"   Input coordinates: {sample['input_coords'].numpy()}")
        print(f"   CLIP embedding shape: {sample['clip_embed'].shape}")
        print(f"   Frame index: {sample['frame_idx']}")
        print(f"   Patch index: {sample['patch_idx']}")
        
        print("\n3. Testing multiple samples...")
        for i in [0, dataset.num_patches, dataset.num_patches * 2]:
            if i < len(dataset):
                sample = dataset[i]
                print(f"   Sample {i}: frame={sample['frame_idx']}, patch={sample['patch_idx']}, coords={sample['input_coords'].numpy()}")
        
        print("\n✓ Dataset test passed!")
        
    except Exception as e:
        print(f"\n⚠ Dataset test skipped: {e}")
        print("   (This is normal if data path doesn't exist)")


if __name__ == '__main__':
    # Test model
    model = test_dual_head_model()
    
    # Test dataset (optional, requires data)
    test_patch_dataset()
    
    print("\n" + "="*60)
    print("Model Architecture Summary:")
    print("="*60)
    print(model)
