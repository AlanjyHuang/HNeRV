import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os

# Get the CSV file path from command line or use default
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    csv_path = 'output/eval_all_patches/per_frame_stats.csv'

if not os.path.exists(csv_path):
    print(f"Error: File not found: {csv_path}")
    print(f"Usage: python {sys.argv[0]} <path_to_per_frame_stats.csv>")
    sys.exit(1)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Read the data
print(f"Reading data from: {csv_path}")
data = pd.read_csv(csv_path)

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 12))

# 1. PSNR over frames
ax1 = plt.subplot(3, 3, 1)
train_data = data[data['split'] == 'train']
val_data = data[data['split'] == 'val']

ax1.plot(train_data['frame_idx'], train_data['mean_psnr'], 'o-', label='Train', alpha=0.7, markersize=4)
ax1.fill_between(train_data['frame_idx'], 
                  train_data['mean_psnr'] - train_data['std_psnr'],
                  train_data['mean_psnr'] + train_data['std_psnr'],
                  alpha=0.2)
ax1.plot(val_data['frame_idx'], val_data['mean_psnr'], 's-', label='Val', alpha=0.7, markersize=6)
ax1.fill_between(val_data['frame_idx'],
                  val_data['mean_psnr'] - val_data['std_psnr'],
                  val_data['mean_psnr'] + val_data['std_psnr'],
                  alpha=0.2)
ax1.set_xlabel('Frame Index')
ax1.set_ylabel('PSNR (dB)')
ax1.set_title('PSNR over Frames')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. SSIM over frames
ax2 = plt.subplot(3, 3, 2)
ax2.plot(train_data['frame_idx'], train_data['mean_ssim'], 'o-', label='Train', alpha=0.7, markersize=4)
ax2.fill_between(train_data['frame_idx'],
                  train_data['mean_ssim'] - train_data['std_ssim'],
                  train_data['mean_ssim'] + train_data['std_ssim'],
                  alpha=0.2)
ax2.plot(val_data['frame_idx'], val_data['mean_ssim'], 's-', label='Val', alpha=0.7, markersize=6)
ax2.fill_between(val_data['frame_idx'],
                  val_data['mean_ssim'] - val_data['std_ssim'],
                  val_data['mean_ssim'] + val_data['std_ssim'],
                  alpha=0.2)
ax2.set_xlabel('Frame Index')
ax2.set_ylabel('SSIM')
ax2.set_title('SSIM over Frames')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. CLIP Similarity over frames
ax3 = plt.subplot(3, 3, 3)
ax3.plot(train_data['frame_idx'], train_data['mean_clip_similarity'], 'o-', label='Train', alpha=0.7, markersize=4)
ax3.fill_between(train_data['frame_idx'],
                  train_data['mean_clip_similarity'] - train_data['std_clip_similarity'],
                  train_data['mean_clip_similarity'] + train_data['std_clip_similarity'],
                  alpha=0.2)
ax3.plot(val_data['frame_idx'], val_data['mean_clip_similarity'], 's-', label='Val', alpha=0.7, markersize=6)
ax3.fill_between(val_data['frame_idx'],
                  val_data['mean_clip_similarity'] - val_data['std_clip_similarity'],
                  val_data['mean_clip_similarity'] + val_data['std_clip_similarity'],
                  alpha=0.2)
ax3.set_xlabel('Frame Index')
ax3.set_ylabel('CLIP Similarity')
ax3.set_title('CLIP Similarity over Frames')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Distribution comparison - PSNR
ax4 = plt.subplot(3, 3, 4)
ax4.boxplot([train_data['mean_psnr'], val_data['mean_psnr']], 
            labels=['Train', 'Val'],
            patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7),
            medianprops=dict(color='red', linewidth=2))
ax4.set_ylabel('PSNR (dB)')
ax4.set_title('PSNR Distribution')
ax4.grid(True, alpha=0.3, axis='y')

# 5. Distribution comparison - SSIM
ax5 = plt.subplot(3, 3, 5)
ax5.boxplot([train_data['mean_ssim'], val_data['mean_ssim']], 
            labels=['Train', 'Val'],
            patch_artist=True,
            boxprops=dict(facecolor='lightgreen', alpha=0.7),
            medianprops=dict(color='red', linewidth=2))
ax5.set_ylabel('SSIM')
ax5.set_title('SSIM Distribution')
ax5.grid(True, alpha=0.3, axis='y')

# 6. Distribution comparison - CLIP
ax6 = plt.subplot(3, 3, 6)
ax6.boxplot([train_data['mean_clip_similarity'], val_data['mean_clip_similarity']], 
            labels=['Train', 'Val'],
            patch_artist=True,
            boxprops=dict(facecolor='lightcoral', alpha=0.7),
            medianprops=dict(color='red', linewidth=2))
ax6.set_ylabel('CLIP Similarity')
ax6.set_title('CLIP Similarity Distribution')
ax6.grid(True, alpha=0.3, axis='y')

# 7. Min-Max range for PSNR
ax7 = plt.subplot(3, 3, 7)
ax7.plot(train_data['frame_idx'], train_data['min_psnr'], 'v', label='Train Min', alpha=0.5, markersize=3)
ax7.plot(train_data['frame_idx'], train_data['max_psnr'], '^', label='Train Max', alpha=0.5, markersize=3)
ax7.plot(val_data['frame_idx'], val_data['min_psnr'], 'v', label='Val Min', alpha=0.7, markersize=5)
ax7.plot(val_data['frame_idx'], val_data['max_psnr'], '^', label='Val Max', alpha=0.7, markersize=5)
ax7.set_xlabel('Frame Index')
ax7.set_ylabel('PSNR (dB)')
ax7.set_title('PSNR Min-Max Range')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# 8. Standard deviation comparison
ax8 = plt.subplot(3, 3, 8)
width = 0.25
x = np.arange(2)
ax8.bar(x - width, [train_data['std_psnr'].mean(), val_data['std_psnr'].mean()], 
        width, label='PSNR', alpha=0.7)
ax8.bar(x, [train_data['std_ssim'].mean() * 100, val_data['std_ssim'].mean() * 100], 
        width, label='SSIM (×100)', alpha=0.7)
ax8.bar(x + width, [train_data['std_clip_similarity'].mean() * 1000, 
                     val_data['std_clip_similarity'].mean() * 1000], 
        width, label='CLIP (×1000)', alpha=0.7)
ax8.set_ylabel('Standard Deviation')
ax8.set_title('Average Std Dev Comparison')
ax8.set_xticks(x)
ax8.set_xticklabels(['Train', 'Val'])
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# 9. Correlation heatmap
ax9 = plt.subplot(3, 3, 9)
metrics_train = train_data[['mean_psnr', 'mean_ssim', 'mean_clip_similarity']].corr()
im = ax9.imshow(metrics_train, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax9.set_xticks(range(3))
ax9.set_yticks(range(3))
ax9.set_xticklabels(['PSNR', 'SSIM', 'CLIP'], rotation=45)
ax9.set_yticklabels(['PSNR', 'SSIM', 'CLIP'])
ax9.set_title('Metric Correlation (Train)')
# Add correlation values
for i in range(3):
    for j in range(3):
        text = ax9.text(j, i, f'{metrics_train.iloc[i, j]:.3f}',
                       ha="center", va="center", color="black", fontsize=10)
plt.colorbar(im, ax=ax9)

plt.tight_layout()
plt.savefig('evaluation_results_comprehensive.png', dpi=300, bbox_inches='tight')
print("Saved: evaluation_results_comprehensive.png")

# Create a second figure with summary statistics
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Summary statistics table
summary_stats = pd.DataFrame({
    'Metric': ['PSNR (dB)', 'SSIM', 'CLIP Similarity'],
    'Train Mean': [
        train_data['mean_psnr'].mean(),
        train_data['mean_ssim'].mean(),
        train_data['mean_clip_similarity'].mean()
    ],
    'Train Std': [
        train_data['mean_psnr'].std(),
        train_data['mean_ssim'].std(),
        train_data['mean_clip_similarity'].std()
    ],
    'Val Mean': [
        val_data['mean_psnr'].mean(),
        val_data['mean_ssim'].mean(),
        val_data['mean_clip_similarity'].mean()
    ],
    'Val Std': [
        val_data['mean_psnr'].std(),
        val_data['mean_ssim'].std(),
        val_data['mean_clip_similarity'].std()
    ]
})

# Plot 1: Overall comparison
ax = axes[0, 0]
x_pos = np.arange(len(summary_stats))
width = 0.35
ax.bar(x_pos - width/2, summary_stats['Train Mean'], width, 
       label='Train', alpha=0.7, yerr=summary_stats['Train Std'], capsize=5)
ax.bar(x_pos + width/2, summary_stats['Val Mean'], width,
       label='Val', alpha=0.7, yerr=summary_stats['Val Std'], capsize=5)
ax.set_ylabel('Mean Value')
ax.set_title('Overall Metric Comparison')
ax.set_xticks(x_pos)
ax.set_xticklabels(summary_stats['Metric'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Train vs Val Gap
ax = axes[0, 1]
gap = summary_stats['Train Mean'] - summary_stats['Val Mean']
colors = ['green' if g > 0 else 'red' for g in gap]
ax.bar(range(len(gap)), gap, color=colors, alpha=0.7)
ax.set_ylabel('Train - Val (Mean)')
ax.set_title('Train/Val Performance Gap')
ax.set_xticks(range(len(gap)))
ax.set_xticklabels(summary_stats['Metric'])
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Temporal stability (std over frames)
ax = axes[1, 0]
ax.plot(train_data['frame_idx'], train_data['std_psnr'], 'o-', label='PSNR Std', alpha=0.7, markersize=3)
ax.plot(train_data['frame_idx'], train_data['std_ssim'] * 50, 's-', label='SSIM Std (×50)', alpha=0.7, markersize=3)
ax.plot(train_data['frame_idx'], train_data['std_clip_similarity'] * 500, '^-', label='CLIP Std (×500)', alpha=0.7, markersize=3)
ax.set_xlabel('Frame Index')
ax.set_ylabel('Standard Deviation')
ax.set_title('Temporal Stability (Train Set)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Print summary table
ax = axes[1, 1]
ax.axis('tight')
ax.axis('off')
table_data = []
table_data.append(['Metric', 'Train Mean±Std', 'Val Mean±Std', 'Gap'])
for idx, row in summary_stats.iterrows():
    train_str = f"{row['Train Mean']:.4f}±{row['Train Std']:.4f}"
    val_str = f"{row['Val Mean']:.4f}±{row['Val Std']:.4f}"
    gap_str = f"{row['Train Mean'] - row['Val Mean']:.4f}"
    table_data.append([row['Metric'], train_str, val_str, gap_str])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.3, 0.3, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.tight_layout()
plt.savefig('evaluation_summary.png', dpi=300, bbox_inches='tight')
print("Saved: evaluation_summary.png")

# Print statistics to console
print("\n" + "="*60)
print("EVALUATION SUMMARY STATISTICS")
print("="*60)
print(f"\nDataset: {len(train_data)} train frames, {len(val_data)} val frames")
print(f"Total frames: {len(data)}")
print("\n" + "-"*60)
print("PSNR (dB):")
print(f"  Train: {train_data['mean_psnr'].mean():.4f} ± {train_data['mean_psnr'].std():.4f}")
print(f"  Val:   {val_data['mean_psnr'].mean():.4f} ± {val_data['mean_psnr'].std():.4f}")
print(f"  Gap:   {train_data['mean_psnr'].mean() - val_data['mean_psnr'].mean():.4f} dB")
print("\n" + "-"*60)
print("SSIM:")
print(f"  Train: {train_data['mean_ssim'].mean():.6f} ± {train_data['mean_ssim'].std():.6f}")
print(f"  Val:   {val_data['mean_ssim'].mean():.6f} ± {val_data['mean_ssim'].std():.6f}")
print(f"  Gap:   {train_data['mean_ssim'].mean() - val_data['mean_ssim'].mean():.6f}")
print("\n" + "-"*60)
print("CLIP Similarity:")
print(f"  Train: {train_data['mean_clip_similarity'].mean():.6f} ± {train_data['mean_clip_similarity'].std():.6f}")
print(f"  Val:   {val_data['mean_clip_similarity'].mean():.6f} ± {val_data['mean_clip_similarity'].std():.6f}")
print(f"  Gap:   {train_data['mean_clip_similarity'].mean() - val_data['mean_clip_similarity'].mean():.6f}")
print("="*60)

plt.show()
