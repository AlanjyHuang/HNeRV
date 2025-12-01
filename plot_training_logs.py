import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_rank0_log(log_file):
    """Parse rank0.txt to extract training metrics"""
    
    metrics = defaultdict(lambda: {'epoch': [], 'value': []})
    
    with open(log_file, 'r') as f:
        for line in f:
            # Parse training lines
            # Format: [2025/11/21 02:57:56] Rank:0, Epoch[10/300], Step [1/77], lr:3.70e-04 pred_PSNR: 17.35 pixel_loss: 0.1675 clip_loss: 0.0000 total_loss: 0.1675
            train_match = re.search(r'Epoch\[(\d+)/\d+\].*lr:([\d.e-]+).*pred_PSNR:\s*([\d.]+).*pixel_loss:\s*([\d.]+).*clip_loss:\s*([\d.]+).*total_loss:\s*([\d.]+)', line)
            
            if train_match:
                epoch = int(train_match.group(1))
                lr = float(train_match.group(2))
                psnr = float(train_match.group(3))
                pixel_loss = float(train_match.group(4))
                clip_loss = float(train_match.group(5))
                total_loss = float(train_match.group(6))
                
                # Only keep last value per epoch (end of epoch)
                if not metrics['lr']['epoch'] or metrics['lr']['epoch'][-1] != epoch:
                    metrics['lr']['epoch'].append(epoch)
                    metrics['lr']['value'].append(lr)
                    metrics['psnr']['epoch'].append(epoch)
                    metrics['psnr']['value'].append(psnr)
                    metrics['pixel_loss']['epoch'].append(epoch)
                    metrics['pixel_loss']['value'].append(pixel_loss)
                    metrics['clip_loss']['epoch'].append(epoch)
                    metrics['clip_loss']['value'].append(clip_loss)
                    metrics['total_loss']['epoch'].append(epoch)
                    metrics['total_loss']['value'].append(total_loss)
                else:
                    # Update with latest value for this epoch
                    metrics['lr']['value'][-1] = lr
                    metrics['psnr']['value'][-1] = psnr
                    metrics['pixel_loss']['value'][-1] = pixel_loss
                    metrics['clip_loss']['value'][-1] = clip_loss
                    metrics['total_loss']['value'][-1] = total_loss
    
    return metrics

def plot_training_metrics(metrics, save_path='training_metrics.png'):
    """Create comprehensive training plots"""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Training Metrics Over Epochs', fontsize=16, fontweight='bold', y=0.995)
    
    # Plot 1: Learning Rate
    ax = axes[0, 0]
    if metrics['lr']['epoch']:
        ax.plot(metrics['lr']['epoch'], metrics['lr']['value'], 'b-', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Learning Rate', fontsize=11)
        ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # Plot 2: PSNR
    ax = axes[0, 1]
    if metrics['psnr']['epoch']:
        ax.plot(metrics['psnr']['epoch'], metrics['psnr']['value'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('PSNR (dB)', fontsize=11)
        ax.set_title('Peak Signal-to-Noise Ratio', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # Add final value annotation
        final_psnr = metrics['psnr']['value'][-1]
        ax.axhline(y=final_psnr, color='g', linestyle='--', alpha=0.5, 
                   label=f'Final: {final_psnr:.2f} dB')
        ax.legend()
    
    # Plot 3: Pixel Loss
    ax = axes[1, 0]
    if metrics['pixel_loss']['epoch']:
        ax.plot(metrics['pixel_loss']['epoch'], metrics['pixel_loss']['value'], 
                'purple', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Pixel Loss', fontsize=11)
        ax.set_title('Pixel Reconstruction Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: CLIP Loss
    ax = axes[1, 1]
    if metrics['clip_loss']['epoch']:
        ax.plot(metrics['clip_loss']['epoch'], metrics['clip_loss']['value'], 
                'orange', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('CLIP Loss', fontsize=11)
        ax.set_title('CLIP Embedding Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # Mark warmup period (assuming 50 epochs)
        if len(metrics['clip_loss']['epoch']) > 0:
            warmup_epochs = 50
            ax.axvline(x=warmup_epochs, color='red', linestyle='--', alpha=0.5,
                      label=f'Warmup end (epoch {warmup_epochs})')
            ax.legend()
    
    # Plot 5: Total Loss
    ax = axes[2, 0]
    if metrics['total_loss']['epoch']:
        ax.plot(metrics['total_loss']['epoch'], metrics['total_loss']['value'], 
                'red', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Total Loss', fontsize=11)
        ax.set_title('Total Loss (Pixel + CLIP)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Plot 6: Combined Loss Comparison
    ax = axes[2, 1]
    if metrics['pixel_loss']['epoch']:
        ax.plot(metrics['pixel_loss']['epoch'], metrics['pixel_loss']['value'], 
                'purple', linewidth=2, label='Pixel Loss', alpha=0.7)
    if metrics['clip_loss']['epoch']:
        ax.plot(metrics['clip_loss']['epoch'], metrics['clip_loss']['value'], 
                'orange', linewidth=2, label='CLIP Loss', alpha=0.7)
    if metrics['total_loss']['epoch']:
        ax.plot(metrics['total_loss']['epoch'], metrics['total_loss']['value'], 
                'red', linewidth=2, label='Total Loss', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss Value', fontsize=11)
    ax.set_title('Loss Components Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    
    return fig

def print_summary(metrics):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    if metrics['psnr']['epoch']:
        print(f"\nTotal epochs: {max(metrics['psnr']['epoch'])}")
        print(f"\nFinal PSNR: {metrics['psnr']['value'][-1]:.2f} dB")
        print(f"Max PSNR: {max(metrics['psnr']['value']):.2f} dB (epoch {metrics['psnr']['epoch'][np.argmax(metrics['psnr']['value'])]})")
    
    if metrics['pixel_loss']['epoch']:
        print(f"\nFinal Pixel Loss: {metrics['pixel_loss']['value'][-1]:.6f}")
        print(f"Min Pixel Loss: {min(metrics['pixel_loss']['value']):.6f} (epoch {metrics['pixel_loss']['epoch'][np.argmin(metrics['pixel_loss']['value'])]})")
    
    if metrics['clip_loss']['epoch']:
        # Find when CLIP loss starts (non-zero)
        nonzero_clip = [(e, v) for e, v in zip(metrics['clip_loss']['epoch'], metrics['clip_loss']['value']) if v > 0]
        if nonzero_clip:
            print(f"\nCLIP loss activated at epoch: {nonzero_clip[0][0]}")
            print(f"Final CLIP Loss: {metrics['clip_loss']['value'][-1]:.6f}")
            clip_values = [v for v in metrics['clip_loss']['value'] if v > 0]
            if clip_values:
                print(f"Min CLIP Loss (after warmup): {min(clip_values):.6f}")
    
    if metrics['total_loss']['epoch']:
        print(f"\nFinal Total Loss: {metrics['total_loss']['value'][-1]:.6f}")
        print(f"Min Total Loss: {min(metrics['total_loss']['value']):.6f} (epoch {metrics['total_loss']['epoch'][np.argmin(metrics['total_loss']['value'])]})")
    
    if metrics['lr']['epoch']:
        print(f"\nInitial Learning Rate: {metrics['lr']['value'][0]:.2e}")
        print(f"Final Learning Rate: {metrics['lr']['value'][-1]:.2e}")
    
    print("="*60 + "\n")

# Main execution
if __name__ == "__main__":
    import sys
    
    # Get log file path from command line or use default
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = 'rank0.txt'
    
    print(f"Parsing log file: {log_file}")
    
    try:
        metrics = parse_rank0_log(log_file)
        
        if not metrics['psnr']['epoch']:
            print("Warning: No metrics found in log file. Check the file format.")
        else:
            print(f"Found {len(metrics['psnr']['epoch'])} epochs of data")
            
            # Print summary
            print_summary(metrics)
            
            # Create plots
            plot_training_metrics(metrics)
            
            plt.show()
            
    except FileNotFoundError:
        print(f"Error: Could not find log file '{log_file}'")
        print(f"Usage: python plot_training_logs.py [path_to_rank0.txt]")
