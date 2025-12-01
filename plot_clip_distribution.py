import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Read the CLIP similarity results
df = pd.read_csv('clip_similarity_corrected.csv')

# Create figure with multiple plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('CLIP Similarity Distribution Analysis', fontsize=16, fontweight='bold')

# Separate train and validation
train_data = df[df['Is_Train_Frame'] == True]['CLIP_Similarity']
val_data = df[df['Is_Train_Frame'] == False]['CLIP_Similarity']

# Plot 1: Overall Distribution (Histogram)
ax1 = axes[0, 0]
bins = np.linspace(0.93, 1.0, 40)
ax1.hist(train_data, bins=bins, alpha=0.6, label='Train', color='blue', edgecolor='black', density=True)
ax1.hist(val_data, bins=bins, alpha=0.6, label='Validation', color='red', edgecolor='black', density=True)
ax1.axvline(train_data.mean(), color='blue', linestyle='--', linewidth=2, 
            label=f'Train Mean: {train_data.mean():.4f}')
ax1.axvline(val_data.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Val Mean: {val_data.mean():.4f}')
ax1.set_xlabel('CLIP Similarity', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('CLIP Similarity Distribution (Normalized)', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot 2: Violin Plot
ax2 = axes[0, 1]
data_for_violin = [train_data.values, val_data.values]
parts = ax2.violinplot(data_for_violin, positions=[1, 2], showmeans=True, showmedians=True)
ax2.set_xticks([1, 2])
ax2.set_xticklabels(['Train', 'Validation'])
ax2.set_ylabel('CLIP Similarity', fontsize=12)
ax2.set_title('CLIP Similarity Distribution (Violin Plot)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
# Add statistics text
stats_text = f"Train: μ={train_data.mean():.4f}, σ={train_data.std():.4f}\n"
stats_text += f"Val: μ={val_data.mean():.4f}, σ={val_data.std():.4f}"
ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 3: Box Plot with individual points
ax3 = axes[1, 0]
bp = ax3.boxplot([train_data.values, val_data.values], 
                  labels=['Train', 'Validation'],
                  patch_artist=True,
                  showfliers=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax3.set_ylabel('CLIP Similarity', fontsize=12)
ax3.set_title('CLIP Similarity Box Plot', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add percentile lines
train_p25, train_p75 = np.percentile(train_data, [25, 75])
val_p25, val_p75 = np.percentile(val_data, [25, 75])
stats_box = f"Train IQR: [{train_p25:.4f}, {train_p75:.4f}]\n"
stats_box += f"Val IQR: [{val_p25:.4f}, {val_p75:.4f}]"
ax3.text(0.02, 0.02, stats_box, transform=ax3.transAxes, 
         fontsize=10, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 4: Cumulative Distribution Function (CDF)
ax4 = axes[1, 1]
train_sorted = np.sort(train_data)
val_sorted = np.sort(val_data)
train_cdf = np.arange(1, len(train_sorted) + 1) / len(train_sorted)
val_cdf = np.arange(1, len(val_sorted) + 1) / len(val_sorted)

ax4.plot(train_sorted, train_cdf, 'b-', linewidth=2, label='Train', alpha=0.7)
ax4.plot(val_sorted, val_cdf, 'r-', linewidth=2, label='Validation', alpha=0.7)
ax4.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Median')
ax4.set_xlabel('CLIP Similarity', fontsize=12)
ax4.set_ylabel('Cumulative Probability', fontsize=12)
ax4.set_title('Cumulative Distribution Function', fontsize=13, fontweight='bold')
ax4.legend(loc='lower right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('clip_similarity_distribution.png', dpi=300, bbox_inches='tight')
print("Plot saved to: clip_similarity_distribution.png")

# Print detailed statistics
print("\n" + "="*70)
print("CLIP SIMILARITY STATISTICS")
print("="*70)

print(f"\nTrain Set ({len(train_data)} frames):")
print(f"  Mean:     {train_data.mean():.6f}")
print(f"  Median:   {train_data.median():.6f}")
print(f"  Std Dev:  {train_data.std():.6f}")
print(f"  Min:      {train_data.min():.6f}")
print(f"  Max:      {train_data.max():.6f}")
print(f"  25th %:   {np.percentile(train_data, 25):.6f}")
print(f"  75th %:   {np.percentile(train_data, 75):.6f}")

print(f"\nValidation Set ({len(val_data)} frames):")
print(f"  Mean:     {val_data.mean():.6f}")
print(f"  Median:   {val_data.median():.6f}")
print(f"  Std Dev:  {val_data.std():.6f}")
print(f"  Min:      {val_data.min():.6f}")
print(f"  Max:      {val_data.max():.6f}")
print(f"  25th %:   {np.percentile(val_data, 25):.6f}")
print(f"  75th %:   {np.percentile(val_data, 75):.6f}")

print(f"\nDifference (Train - Val):")
print(f"  Mean Δ:   {train_data.mean() - val_data.mean():.6f}")
print(f"  Median Δ: {train_data.median() - val_data.median():.6f}")

print(f"\nPercentage of frames with similarity > 0.99:")
print(f"  Train: {(train_data > 0.99).sum() / len(train_data) * 100:.1f}%")
print(f"  Val:   {(val_data > 0.99).sum() / len(val_data) * 100:.1f}%")

print(f"\nPercentage of frames with similarity > 0.98:")
print(f"  Train: {(train_data > 0.98).sum() / len(train_data) * 100:.1f}%")
print(f"  Val:   {(val_data > 0.98).sum() / len(val_data) * 100:.1f}%")

print("="*70 + "\n")

plt.show()
