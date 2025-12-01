import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV data
df = pd.read_csv('clip_similarity_results.csv')

# Fix the Is_Train_Frame column based on the actual split logic
# For 9_9_10 split:
# valid_train_length = 9, total_train_length = 9, total_data_length = 10
# Train: (frame_idx % 10) < 9  --> frames 0-8, 10-18, 20-28, etc.
# Val:   (frame_idx % 10) >= 9 --> frames 9, 19, 29, etc.
def is_train_frame(frame_idx, valid_train_length=9, total_data_length=10):
    return (frame_idx % total_data_length) < valid_train_length

# Update the column - convert to False for validation frames
df['Is_Train_Frame'] = df['Frame_Index'].apply(is_train_frame)

# Save corrected CSV
df.to_csv('clip_similarity_corrected.csv', index=False)
print("Corrected CSV saved to: clip_similarity_corrected.csv")

# Print summary statistics
train_frames = df[df['Is_Train_Frame'] == True]
val_frames = df[df['Is_Train_Frame'] == False]

print(f"\n=== Summary Statistics ===")
print(f"Total frames: {len(df)}")
print(f"Train frames: {len(train_frames)} (indices: {list(train_frames['Frame_Index'].values)})")
print(f"Val frames: {len(val_frames)} (indices: {list(val_frames['Frame_Index'].values)})")
print(f"\nTrain CLIP similarity: {train_frames['CLIP_Similarity'].mean():.4f} ± {train_frames['CLIP_Similarity'].std():.4f}")
print(f"Val CLIP similarity: {val_frames['CLIP_Similarity'].mean():.4f} ± {val_frames['CLIP_Similarity'].std():.4f}")

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: CLIP similarity over all frames
ax1 = axes[0]
train_indices = train_frames['Frame_Index'].values
train_sims = train_frames['CLIP_Similarity'].values
val_indices = val_frames['Frame_Index'].values
val_sims = val_frames['CLIP_Similarity'].values

ax1.scatter(train_indices, train_sims, c='blue', label='Train', alpha=0.7, s=30)
ax1.scatter(val_indices, val_sims, c='red', label='Validation', alpha=0.7, s=50, marker='s')
ax1.axhline(y=train_frames['CLIP_Similarity'].mean(), color='blue', linestyle='--', 
            label=f'Train Mean: {train_frames["CLIP_Similarity"].mean():.4f}', alpha=0.5)
ax1.axhline(y=val_frames['CLIP_Similarity'].mean(), color='red', linestyle='--', 
            label=f'Val Mean: {val_frames["CLIP_Similarity"].mean():.4f}', alpha=0.5)
ax1.set_xlabel('Frame Index', fontsize=12)
ax1.set_ylabel('CLIP Similarity', fontsize=12)
ax1.set_title('CLIP Embedding Similarity: Train vs Validation Frames', fontsize=14, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.93, 1.0])

# Plot 2: Distribution histogram
ax2 = axes[1]
bins = np.linspace(0.93, 1.0, 30)
ax2.hist(train_sims, bins=bins, alpha=0.6, label='Train', color='blue', edgecolor='black')
ax2.hist(val_sims, bins=bins, alpha=0.6, label='Validation', color='red', edgecolor='black')
ax2.axvline(x=train_frames['CLIP_Similarity'].mean(), color='blue', linestyle='--', linewidth=2, 
            label=f'Train Mean: {train_frames["CLIP_Similarity"].mean():.4f}')
ax2.axvline(x=val_frames['CLIP_Similarity'].mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Val Mean: {val_frames["CLIP_Similarity"].mean():.4f}')
ax2.set_xlabel('CLIP Similarity', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Distribution of CLIP Similarity Scores', fontsize=14, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('clip_similarity_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: clip_similarity_analysis.png")

# Print frames with lowest CLIP similarity
print(f"\n=== Lowest CLIP Similarity Frames ===")
worst_frames = df.nsmallest(10, 'CLIP_Similarity')[['Frame_Index', 'CLIP_Similarity', 'Is_Train_Frame']]
print(worst_frames.to_string(index=False))

plt.show()
