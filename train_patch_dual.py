"""
Training script for Dual-Head Patch-Based HNeRV Model
Input: (frame_idx, patch_x, patch_y)
Output: RGB patch + CLIP embedding
Loss: Hybrid loss = RGB reconstruction loss + CLIP similarity loss
Train/Test split: Frame-based (some frames for training, others for testing)
"""

import imageio
import argparse
import os
import random
import shutil
from datetime import datetime
import numpy as np
import csv
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from model_patch_dual import PatchVideoDataSet, DualHeadHNeRV
from hnerv_utils import *
from torch.utils.data import Subset
from copy import deepcopy
from dahuffman import HuffmanCodec
from torchvision.utils import save_image
import pandas as pd
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='', help='data path for vid')
    parser.add_argument('--vid', type=str, default='k400_train0', help='video id')
    parser.add_argument('--shuffle_data', action='store_true', help='randomly shuffle the frame idx')
    parser.add_argument('--data_split', type=str, default='1_1_1', 
        help='Valid_train/total_train/all data split for FRAMES (not patches)')
    parser.add_argument('--crop_list', type=str, default='640_1280', help='video crop size')
    parser.add_argument('--resize_list', type=str, default='-1', help='video resize size')

    # HNeRV architecture parameters
    parser.add_argument('--embed', type=str, default='pe_1.25_80', help='positional encoding base_levels')
    parser.add_argument('--ks', type=str, default='0_3_3', help='kernel size')
    parser.add_argument('--fc_dim', type=int, default=512, help='FC dimension')
    parser.add_argument('--fc_hw', type=str, default='9_16', help='out size (h,w) for mlp')
    parser.add_argument('--reduce', type=float, default=1.2, help='channel reduction')
    parser.add_argument('--lower_width', type=int, default=32, help='lowest channel width')
    parser.add_argument('--dec_strds', type=int, nargs='+', default=[5, 3, 2, 2, 2], help='decoder strides')
    parser.add_argument('--num_blks', type=str, default='1_1', help='block number')
    parser.add_argument("--conv_type", default=['convnext', 'pshuffel'], type=str, nargs="+", help='conv type')
    parser.add_argument('--norm', default='none', type=str, help='norm layer')
    parser.add_argument('--act', type=str, default='gelu', help='activation')
    parser.add_argument('--clip_dim', type=int, default=512, help='CLIP embedding dimension')

    # Training parameters
    parser.add_argument('-j', '--workers', type=int, help='data loading workers', default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=8, help='batch size')
    parser.add_argument('--start_epoch', type=int, default=-1, help='starting epoch')
    parser.add_argument('--not_resume', action='store_true', help='not resume from checkpoint')
    parser.add_argument('-e', '--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_type', type=str, default='cosine_0.1_1_0.1', help='learning rate schedule')
    parser.add_argument('--loss', type=str, default='L1', help='pixel loss type')
    parser.add_argument('--out_bias', default='tanh', type=str, help='output activation')
    
    # Hybrid loss parameters
    parser.add_argument('--clip_loss_weight', type=float, default=0.1, help='Weight for CLIP loss')
    parser.add_argument('--pixel_loss_warmup_epochs', type=int, default=50, help='Epochs before adding CLIP loss')

    # Evaluation parameters
    parser.add_argument('--eval_only', action='store_true', default=False, help='evaluation only')
    parser.add_argument('--eval_freq', type=int, default=10, help='evaluation frequency')
    parser.add_argument('--dump_images', action='store_true', default=False, help='dump prediction images')
    parser.add_argument('--dump_videos', action='store_true', default=False, help='concat predictions into video')

    # Logging parameters
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('-p', '--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('--weight', default='None', type=str, help='pretrained weights')
    parser.add_argument('--overwrite', action='store_true', help='overwrite output dir')
    parser.add_argument('--outf', default='patch_dual', help='output folder')
    parser.add_argument('--suffix', default='', help='suffix for output folder')
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')

    args = parser.parse_args()
    torch.set_printoptions(precision=4)
    
    if args.debug:
        args.eval_freq = 1
        args.outf = 'output/debug'
    else:
        args.outf = os.path.join('output', args.outf)

    # Build experiment ID
    dec_strd_str = ','.join([str(x) for x in args.dec_strds])
    extra_str = f'DEC_{args.conv_type[1]}_{dec_strd_str}_{args.act}'
    exp_id = f'{args.vid}/{args.data_split}_{args.embed}_FC{args.fc_hw}_KS{args.ks}' + \
             f'_RED{args.reduce}_low{args.lower_width}_blk{args.num_blks}' + \
             f'_e{args.epochs}_b{args.batchSize}_lr{args.lr}_{args.lr_type}_{args.loss}' + \
             f'_CLIP{args.clip_loss_weight}_warmup{args.pixel_loss_warmup_epochs}_{extra_str}{args.suffix}'
    args.exp_id = exp_id
    args.outf = os.path.join(args.outf, exp_id)
    
    if args.overwrite and os.path.isdir(args.outf):
        print('Overwriting existing output dir!')
        shutil.rmtree(args.outf)

    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    torch.set_printoptions(precision=2)
    train(None, args)


def train(local_rank, args):
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup dataset
    full_dataset = PatchVideoDataSet(args)
    args.final_size = full_dataset.final_size
    
    # Frame-based split (not patch-based)
    # Calculate total number of frames
    num_frames = len(full_dataset.video)
    split_num_list = [int(x) for x in args.data_split.split('_')]
    
    # Create frame index lists
    frame_indices = list(range(num_frames))
    train_frame_list, val_frame_list = data_split(frame_indices, split_num_list, args.shuffle_data, 0)
    
    # Convert frame indices to patch indices
    # Each frame has 8 patches (num_patches = 8)
    num_patches_per_frame = full_dataset.num_patches
    train_patch_indices = []
    val_patch_indices = []
    
    for frame_idx in train_frame_list:
        for patch_idx in range(num_patches_per_frame):
            train_patch_indices.append(frame_idx * num_patches_per_frame + patch_idx)
    
    for frame_idx in val_frame_list:
        for patch_idx in range(num_patches_per_frame):
            val_patch_indices.append(frame_idx * num_patches_per_frame + patch_idx)
    
    args.val_frame_list = val_frame_list
    args.train_frame_list = train_frame_list
    
    # Create dataloaders
    train_dataset = Subset(full_dataset, train_patch_indices)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchSize, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn
    )
    
    full_dataloader = torch.utils.data.DataLoader(
        full_dataset, batch_size=args.batchSize, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn
    )

    # Build model
    model = DualHeadHNeRV(args)
    model.to(device)

    # Print model info
    total_param = sum([p.data.nelement() for p in model.parameters()]) / 1e6
    param_str = f'Total_Params_{round(total_param, 2)}M'
    print(f'{args}\n{model}\n{param_str}', flush=True)
    with open('{}/rank0.txt'.format(args.outf), 'a') as f:
        f.write(str(model) + '\n' + f'{param_str}\n')
    
    writer = SummaryWriter(os.path.join(args.outf, param_str, 'tensorboard'))

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.)

    # Resume from checkpoint
    checkpoint = None
    if not args.not_resume:
        checkpoint_path = os.path.join(args.outf, 'model_latest.pth')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")

    if args.start_epoch < 0:
        args.start_epoch = checkpoint['epoch'] if checkpoint is not None else 0

    if args.eval_only:
        print('Evaluation mode...')
        evaluate(model, full_dataset, full_dataloader, device, args, dump_vis=True, epoch=None, writer=None)
        return

    # Training loop
    start = datetime.now()
    psnr_list = []
    
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        epoch_start_time = datetime.now()
        
        # Check if this is the warmup transition epoch
        if epoch == args.pixel_loss_warmup_epochs:
            print("\n" + "="*80)
            print(f"🔥 WARMUP COMPLETE! Starting CLIP loss training at epoch {epoch+1}")
            print(f"   CLIP loss weight: {args.clip_loss_weight}")
            print("="*80 + "\n")
            with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"WARMUP COMPLETE! Starting CLIP loss training at epoch {epoch+1}\n")
                f.write(f"CLIP loss weight: {args.clip_loss_weight}\n")
                f.write(f"{'='*80}\n\n")
        
        pred_psnr_list = []
        pixel_loss_list = []
        clip_loss_list = []
        total_loss_list = []
        grad_norm_list = []
        rgb_grad_norm_list = []
        clip_grad_norm_list = []
        decoder_grad_norm_list = []
        
        for i, sample in enumerate(train_dataloader):
            # Move data to device
            patch_img = sample['img'].to(device)  # [batch, 3, patch_h, patch_w]
            input_coords = sample['input_coords'].to(device)  # [batch, 3] - (t, x, y)
            clip_embed_gt = sample['clip_embed'].to(device)  # [batch, 512]
            
            if i > 10 and args.debug:
                break

            # Forward pass
            cur_epoch = (epoch + float(i) / len(train_dataloader)) / args.epochs
            lr = adjust_lr(optimizer, cur_epoch, args)
            
            # Model outputs: rgb_out, clip_out, embed_list, dec_time
            rgb_out, clip_out, embed_list, dec_time = model(input_coords)
            
            # Resize RGB output to match patch size if needed
            if rgb_out.shape[-2:] != patch_img.shape[-2:]:
                rgb_out = F.interpolate(rgb_out, size=patch_img.shape[-2:], mode='bilinear', align_corners=False)
            
            # Compute pixel loss
            pixel_loss = loss_fn(rgb_out, patch_img, args.loss)
            
            # Compute CLIP loss (cosine similarity)
            clip_loss = torch.tensor(0.0).to(device)
            clip_loss_active = False
            if epoch >= args.pixel_loss_warmup_epochs:
                # Compute cosine similarity loss
                # clip_out: [batch, 512], clip_embed_gt: [batch, 512]
                # Both are already normalized
                clip_sim = F.cosine_similarity(clip_out, clip_embed_gt, dim=-1)
                clip_loss = (1 - clip_sim).mean()
                clip_loss_active = True
            
            # Hybrid loss
            total_loss = pixel_loss + args.clip_loss_weight * clip_loss
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            
            # Compute gradient norms (before optimizer step)
            total_grad_norm = 0.0
            rgb_head_grad_norm = 0.0
            clip_head_grad_norm = 0.0
            decoder_grad_norm = 0.0
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_grad_norm += param_norm ** 2
                    
                    if 'rgb_head' in name:
                        rgb_head_grad_norm += param_norm ** 2
                    elif 'clip_head' in name:
                        clip_head_grad_norm += param_norm ** 2
                    elif 'decoder' in name:
                        decoder_grad_norm += param_norm ** 2
            
            total_grad_norm = total_grad_norm ** 0.5
            rgb_head_grad_norm = rgb_head_grad_norm ** 0.5
            clip_head_grad_norm = clip_head_grad_norm ** 0.5
            decoder_grad_norm = decoder_grad_norm ** 0.5
            
            optimizer.step()

            # Track losses
            pixel_loss_list.append(pixel_loss.item())
            clip_loss_list.append(clip_loss.item())
            total_loss_list.append(total_loss.item())
            pred_psnr_list.append(psnr_fn_single(rgb_out.detach(), patch_img))
            
            # Track gradient norms
            grad_norm_list.append(total_grad_norm)
            rgb_grad_norm_list.append(rgb_head_grad_norm)
            clip_grad_norm_list.append(clip_head_grad_norm)
            decoder_grad_norm_list.append(decoder_grad_norm)
            
            # Print progress
            if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
                pred_psnr = torch.cat(pred_psnr_list).mean()
                avg_pixel_loss = sum(pixel_loss_list) / len(pixel_loss_list)
                avg_clip_loss = sum(clip_loss_list) / len(clip_loss_list)
                avg_total_loss = sum(total_loss_list) / len(total_loss_list)
                
                # Add warmup status indicator
                warmup_status = "WARMUP" if epoch < args.pixel_loss_warmup_epochs else "CLIP_ACTIVE"
                
                print_str = '[{}] Epoch[{}/{}], Step[{}/{}], lr:{:.2e}, PSNR:{}, pixel_loss:{:.4f}, clip_loss:{:.4f}, total_loss:{:.4f} [{}]'.format(
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S"), epoch+1, args.epochs, i+1, len(train_dataloader),
                    lr, RoundTensor(pred_psnr, 2), avg_pixel_loss, avg_clip_loss, avg_total_loss, warmup_status
                )
                print(print_str, flush=True)
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')
                
                # Log batch-level metrics to tensorboard
                global_step = epoch * len(train_dataloader) + i
                writer.add_scalar('Train_Batch/PSNR', pred_psnr.item(), global_step)
                writer.add_scalar('Train_Batch/pixel_loss', avg_pixel_loss, global_step)
                writer.add_scalar('Train_Batch/clip_loss', avg_clip_loss, global_step)
                writer.add_scalar('Train_Batch/total_loss', avg_total_loss, global_step)

        # Log to tensorboard - Training metrics
        pred_psnr = torch.cat(pred_psnr_list).mean()
        epoch_avg_pixel_loss = sum(pixel_loss_list) / len(pixel_loss_list)
        epoch_avg_clip_loss = sum(clip_loss_list) / len(clip_loss_list)
        epoch_avg_total_loss = sum(total_loss_list) / len(total_loss_list)
        
        writer.add_scalar('Train/PSNR', pred_psnr.item(), epoch+1)
        writer.add_scalar('Train/learning_rate', lr, epoch+1)
        writer.add_scalar('Train/Loss/pixel_loss', epoch_avg_pixel_loss, epoch+1)
        writer.add_scalar('Train/Loss/clip_loss', epoch_avg_clip_loss, epoch+1)
        writer.add_scalar('Train/Loss/total_loss', epoch_avg_total_loss, epoch+1)
        writer.add_scalar('Train/Loss/clip_loss_weighted', epoch_avg_clip_loss * args.clip_loss_weight, epoch+1)
        
        # Log loss ratios
        if epoch_avg_clip_loss > 0:
            writer.add_scalar('Train/Loss/pixel_to_clip_ratio', epoch_avg_pixel_loss / epoch_avg_clip_loss, epoch+1)
        
        # Log CLIP loss activation (whether it's being used)
        clip_loss_active = 1.0 if epoch >= args.pixel_loss_warmup_epochs else 0.0
        writer.add_scalar('Train/clip_loss_active', clip_loss_active, epoch+1)
        
        # Log gradient norms
        if grad_norm_list:
            writer.add_scalar('Train/Gradients/total_norm', sum(grad_norm_list) / len(grad_norm_list), epoch+1)
            writer.add_scalar('Train/Gradients/rgb_head_norm', sum(rgb_grad_norm_list) / len(rgb_grad_norm_list), epoch+1)
            writer.add_scalar('Train/Gradients/clip_head_norm', sum(clip_grad_norm_list) / len(clip_grad_norm_list), epoch+1)
            writer.add_scalar('Train/Gradients/decoder_norm', sum(decoder_grad_norm_list) / len(decoder_grad_norm_list), epoch+1)
            
            # Log gradient ratios
            avg_rgb_grad = sum(rgb_grad_norm_list) / len(rgb_grad_norm_list)
            avg_clip_grad = sum(clip_grad_norm_list) / len(clip_grad_norm_list)
            if avg_clip_grad > 0:
                writer.add_scalar('Train/Gradients/rgb_to_clip_ratio', avg_rgb_grad / avg_clip_grad, epoch+1)
        
        epoch_end_time = datetime.now()
        print("Time/epoch: Current:{:.2f}s, Average:{:.2f}s".format(
            (epoch_end_time - epoch_start_time).total_seconds(),
            (epoch_end_time - start).total_seconds() / (epoch + 1 - args.start_epoch)
        ))
        
        # Log model parameter statistics every few epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                # RGB head parameters
                rgb_params = [p for name, p in model.named_parameters() if 'rgb_head' in name]
                if rgb_params:
                    rgb_param_mean = torch.cat([p.flatten() for p in rgb_params]).mean().item()
                    rgb_param_std = torch.cat([p.flatten() for p in rgb_params]).std().item()
                    writer.add_scalar('Model/Params/rgb_head_mean', rgb_param_mean, epoch+1)
                    writer.add_scalar('Model/Params/rgb_head_std', rgb_param_std, epoch+1)
                
                # CLIP head parameters
                clip_params = [p for name, p in model.named_parameters() if 'clip_head' in name]
                if clip_params:
                    clip_param_mean = torch.cat([p.flatten() for p in clip_params]).mean().item()
                    clip_param_std = torch.cat([p.flatten() for p in clip_params]).std().item()
                    writer.add_scalar('Model/Params/clip_head_mean', clip_param_mean, epoch+1)
                    writer.add_scalar('Model/Params/clip_head_std', clip_param_std, epoch+1)

        # Evaluation
        if (epoch + 1) % args.eval_freq == 0 or (args.epochs - epoch) in [1, 3, 5]:
            results = evaluate(model, full_dataset, full_dataloader, device, args,
                             dump_vis=(epoch == args.epochs - 1) and args.dump_images,
                             epoch=epoch, writer=writer)
            
            print_str = f'Eval at epoch {epoch+1}: Train_PSNR={results["train_psnr"]:.2f}, ' + \
                       f'Val_PSNR={results["val_psnr"]:.2f}, Train_CLIP_sim={results["train_clip_sim"]:.4f}, ' + \
                       f'Val_CLIP_sim={results["val_clip_sim"]:.4f}'
            print(print_str, flush=True)
            with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')
            
            psnr_list.append(results["val_psnr"])

        # Save checkpoint
        save_checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.outf))
        
        if (epoch + 1) % args.epochs == 0:
            torch.save(save_checkpoint, f'{args.outf}/epoch{epoch+1}.pth')

    print(f"Training complete in: {str(datetime.now() - start)}")
    writer.close()


@torch.no_grad()
def evaluate(model, full_dataset, full_dataloader, device, args, dump_vis=False, epoch=None, writer=None):
    """
    Evaluate the model on all patches and compute per-frame metrics.
    
    TensorBoard Logging:
    - Eval/Frame/: Frame-level PSNR and CLIP similarity metrics
    - Eval/PSNR_Stats/: PSNR statistics (std, min, max)
    - Eval/Patch/: Patch-level statistics
    - Eval/Gap/: Generalization gap (train vs val)
    - Eval/Histogram/: Distribution histograms
    - Eval/Info/: Dataset split information
    """
    model.eval()
    
    # Collect predictions and ground truth per patch
    all_results = {}  # frame_idx -> {'patches': {patch_idx: {'rgb_pred', 'rgb_gt', 'clip_pred', 'clip_gt'}}}
    
    for i, sample in enumerate(full_dataloader):
        patch_img = sample['img'].to(device)
        input_coords = sample['input_coords'].to(device)
        clip_embed_gt = sample['clip_embed'].to(device)
        frame_indices = sample['frame_idx'].cpu().numpy()
        patch_indices = sample['patch_idx'].cpu().numpy()
        
        if i > 10 and args.debug:
            # In debug mode, break early but warn if no val frames seen
            frames_seen = set(sample['frame_idx'].cpu().numpy())
            val_frames_seen = frames_seen.intersection(args.val_frame_list)
            if not val_frames_seen:
                print(f"  [DEBUG] Processed {i+1} batches but no validation frames yet. Continuing a bit more...")
                if i > 50:  # Hard limit
                    break
            else:
                break
        
        # Forward pass
        rgb_out, clip_out, _, _ = model(input_coords)
        
        # Resize RGB output to match patch size if needed
        if rgb_out.shape[-2:] != patch_img.shape[-2:]:
            rgb_out = F.interpolate(rgb_out, size=patch_img.shape[-2:], mode='bilinear', align_corners=False)
        
        # Store results
        for b in range(rgb_out.shape[0]):
            frame_idx = int(frame_indices[b])
            patch_idx = int(patch_indices[b])
            
            if frame_idx not in all_results:
                all_results[frame_idx] = {'patches': {}}
            
            all_results[frame_idx]['patches'][patch_idx] = {
                'rgb_pred': rgb_out[b].cpu(),
                'rgb_gt': patch_img[b].cpu(),
                'clip_pred': clip_out[b].cpu(),
                'clip_gt': clip_embed_gt[b].cpu(),
            }
    
    # Compute metrics per frame
    train_psnr_list = []
    val_psnr_list = []
    train_clip_sim_list = []
    val_clip_sim_list = []
    train_clip_dist_list = []  # CLIP distance (1 - similarity)
    val_clip_dist_list = []
    
    # For patch-level statistics
    all_train_patch_psnr = []
    all_val_patch_psnr = []
    all_train_patch_clip_sim = []
    all_val_patch_clip_sim = []
    
    for frame_idx, frame_data in all_results.items():
        patches = frame_data['patches']
        
        # Compute average PSNR across all patches in this frame
        frame_psnr_list = []
        frame_clip_sim_list = []
        
        for patch_idx, patch_data in patches.items():
            # PSNR
            psnr = psnr_fn_single(patch_data['rgb_pred'].unsqueeze(0), patch_data['rgb_gt'].unsqueeze(0))
            frame_psnr_list.append(psnr.item())
            
            # CLIP similarity
            clip_sim = F.cosine_similarity(
                patch_data['clip_pred'].unsqueeze(0),
                patch_data['clip_gt'].unsqueeze(0),
                dim=-1
            ).item()
            frame_clip_sim_list.append(clip_sim)
        
        frame_avg_psnr = np.mean(frame_psnr_list)
        frame_avg_clip_sim = np.mean(frame_clip_sim_list)
        frame_avg_clip_dist = 1 - frame_avg_clip_sim
        
        # Determine if train or val frame
        if frame_idx in args.train_frame_list:
            train_psnr_list.append(frame_avg_psnr)
            train_clip_sim_list.append(frame_avg_clip_sim)
            train_clip_dist_list.append(frame_avg_clip_dist)
            all_train_patch_psnr.extend(frame_psnr_list)
            all_train_patch_clip_sim.extend(frame_clip_sim_list)
        elif frame_idx in args.val_frame_list:
            val_psnr_list.append(frame_avg_psnr)
            val_clip_sim_list.append(frame_avg_clip_sim)
            val_clip_dist_list.append(frame_avg_clip_dist)
            all_val_patch_psnr.extend(frame_psnr_list)
            all_val_patch_clip_sim.extend(frame_clip_sim_list)
    
    # Compute overall metrics
    results = {
        'train_psnr': np.mean(train_psnr_list) if train_psnr_list else 0,
        'val_psnr': np.mean(val_psnr_list) if val_psnr_list else 0,
        'train_clip_sim': np.mean(train_clip_sim_list) if train_clip_sim_list else 0,
        'val_clip_sim': np.mean(val_clip_sim_list) if val_clip_sim_list else 0,
        'train_clip_dist': np.mean(train_clip_dist_list) if train_clip_dist_list else 0,
        'val_clip_dist': np.mean(val_clip_dist_list) if val_clip_dist_list else 0,
        # Statistics
        'train_psnr_std': np.std(train_psnr_list) if train_psnr_list else 0,
        'val_psnr_std': np.std(val_psnr_list) if val_psnr_list else 0,
        'train_psnr_min': np.min(train_psnr_list) if train_psnr_list else 0,
        'train_psnr_max': np.max(train_psnr_list) if train_psnr_list else 0,
        'val_psnr_min': np.min(val_psnr_list) if val_psnr_list else 0,
        'val_psnr_max': np.max(val_psnr_list) if val_psnr_list else 0,
        # Patch-level statistics
        'train_patch_psnr_std': np.std(all_train_patch_psnr) if all_train_patch_psnr else 0,
        'val_patch_psnr_std': np.std(all_val_patch_psnr) if all_val_patch_psnr else 0,
        'train_patch_clip_sim_std': np.std(all_train_patch_clip_sim) if all_train_patch_clip_sim else 0,
        'val_patch_clip_sim_std': np.std(all_val_patch_clip_sim) if all_val_patch_clip_sim else 0,
    }
    
    # Debug info
    if args.debug:
        print(f"\n[DEBUG] Evaluation summary:")
        print(f"  Total frames processed: {len(all_results)}")
        print(f"  Train frames found: {len(train_psnr_list)} (expected: {len(args.train_frame_list)})")
        print(f"  Val frames found: {len(val_psnr_list)} (expected: {len(args.val_frame_list)})")
        if len(val_psnr_list) == 0:
            print(f"  ⚠️  WARNING: No validation frames were evaluated!")
            print(f"  First few expected val frames: {args.val_frame_list[:5]}")
            print(f"  Frames actually processed: {sorted(all_results.keys())[:10]}")
            print(f"  → Debug mode only processes first 10 batches, may not reach val frames")
            print(f"  → Run without --debug or increase debug limit to evaluate val frames\n")
    
    # Log to tensorboard with comprehensive metrics
    if writer is not None and epoch is not None:
        # === Frame-level metrics ===
        writer.add_scalar('Eval/Frame/train_psnr', results['train_psnr'], epoch+1)
        writer.add_scalar('Eval/Frame/val_psnr', results['val_psnr'], epoch+1)
        writer.add_scalar('Eval/Frame/train_clip_sim', results['train_clip_sim'], epoch+1)
        writer.add_scalar('Eval/Frame/val_clip_sim', results['val_clip_sim'], epoch+1)
        writer.add_scalar('Eval/Frame/train_clip_dist', results['train_clip_dist'], epoch+1)
        writer.add_scalar('Eval/Frame/val_clip_dist', results['val_clip_dist'], epoch+1)
        
        # === PSNR statistics ===
        writer.add_scalar('Eval/PSNR_Stats/train_std', results['train_psnr_std'], epoch+1)
        writer.add_scalar('Eval/PSNR_Stats/val_std', results['val_psnr_std'], epoch+1)
        writer.add_scalar('Eval/PSNR_Stats/train_min', results['train_psnr_min'], epoch+1)
        writer.add_scalar('Eval/PSNR_Stats/train_max', results['train_psnr_max'], epoch+1)
        writer.add_scalar('Eval/PSNR_Stats/val_min', results['val_psnr_min'], epoch+1)
        writer.add_scalar('Eval/PSNR_Stats/val_max', results['val_psnr_max'], epoch+1)
        
        # === Patch-level statistics ===
        writer.add_scalar('Eval/Patch/train_psnr_std', results['train_patch_psnr_std'], epoch+1)
        writer.add_scalar('Eval/Patch/val_psnr_std', results['val_patch_psnr_std'], epoch+1)
        writer.add_scalar('Eval/Patch/train_clip_sim_std', results['train_patch_clip_sim_std'], epoch+1)
        writer.add_scalar('Eval/Patch/val_clip_sim_std', results['val_patch_clip_sim_std'], epoch+1)
        
        # === Generalization gap ===
        psnr_gap = results['train_psnr'] - results['val_psnr']
        clip_sim_gap = results['train_clip_sim'] - results['val_clip_sim']
        writer.add_scalar('Eval/Gap/psnr_gap', psnr_gap, epoch+1)
        writer.add_scalar('Eval/Gap/clip_sim_gap', clip_sim_gap, epoch+1)
        
        # === Histograms ===
        if train_psnr_list:
            writer.add_histogram('Eval/Histogram/train_psnr_distribution', np.array(train_psnr_list), epoch+1)
        if val_psnr_list:
            writer.add_histogram('Eval/Histogram/val_psnr_distribution', np.array(val_psnr_list), epoch+1)
        if train_clip_sim_list:
            writer.add_histogram('Eval/Histogram/train_clip_sim_distribution', np.array(train_clip_sim_list), epoch+1)
        if val_clip_sim_list:
            writer.add_histogram('Eval/Histogram/val_clip_sim_distribution', np.array(val_clip_sim_list), epoch+1)
        
        # === Data split info ===
        writer.add_scalar('Eval/Info/num_train_frames', len(train_psnr_list), epoch+1)
        writer.add_scalar('Eval/Info/num_val_frames', len(val_psnr_list), epoch+1)
        writer.add_scalar('Eval/Info/num_train_patches', len(all_train_patch_psnr), epoch+1)
        writer.add_scalar('Eval/Info/num_val_patches', len(all_val_patch_psnr), epoch+1)
    
    # Dump visualizations
    if dump_vis:
        visual_dir = f'{args.outf}/visualize_patches'
        if not os.path.isdir(visual_dir):
            os.makedirs(visual_dir)
        
        # Save some sample patches
        num_samples = min(20, len(all_results))
        sample_frames = sorted(all_results.keys())[:num_samples]
        
        for frame_idx in sample_frames:
            frame_data = all_results[frame_idx]
            for patch_idx, patch_data in frame_data['patches'].items():
                concat_img = torch.cat([patch_data['rgb_gt'], patch_data['rgb_pred']], dim=2)
                save_image(concat_img, f'{visual_dir}/frame{frame_idx:04d}_patch{patch_idx}.png')
    
    model.train()
    return results


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn', force=True)
    main()
