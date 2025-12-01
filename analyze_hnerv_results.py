import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV data (paste your data here or read from file)
data = """Frame_Index,CLIP_Similarity,Is_Train_Frame
0,0.987949,True
1,0.993010,True
2,0.993252,True
3,0.993050,True
4,0.991758,True
5,0.994055,True
6,0.994581,True
7,0.994183,True
8,0.993557,True
9,0.990448,True
10,0.994361,True
11,0.993959,True
12,0.993721,True
13,0.993362,True
14,0.989387,True
15,0.993406,True
16,0.993972,True
17,0.993576,True
18,0.993018,True
19,0.990979,True
20,0.993352,True
21,0.993753,True
22,0.993426,True
23,0.993248,True
24,0.989172,True
25,0.993877,True
26,0.993516,True
27,0.993670,True
28,0.992602,True
29,0.991193,True
30,0.993380,True
31,0.994134,True
32,0.993643,True
33,0.993349,True
34,0.988656,True
35,0.993058,True
36,0.993785,True
37,0.994050,True
38,0.993504,True
39,0.990811,True
40,0.992505,True
41,0.992448,True
42,0.992945,True
43,0.991894,True
44,0.989866,True
45,0.993537,True
46,0.993148,True
47,0.992072,True
48,0.991801,True
49,0.985874,True
50,0.991664,True
51,0.992480,True
52,0.991744,True
53,0.992595,True
54,0.989921,True
55,0.993138,True
56,0.992926,True
57,0.991430,True
58,0.992395,True
59,0.978546,True
60,0.990491,True
61,0.993416,True
62,0.993677,True
63,0.992393,True
64,0.990299,True
65,0.993461,True
66,0.994159,True
67,0.993662,True
68,0.993812,True
69,0.989930,True
70,0.993973,True
71,0.993813,True
72,0.992847,True
73,0.993284,True
74,0.991412,True
75,0.992812,True
76,0.992348,True
77,0.993338,True
78,0.993798,True
79,0.991630,True
80,0.993283,True
81,0.993156,True
82,0.993598,True
83,0.993291,True
84,0.990241,True
85,0.993333,True
86,0.993822,True
87,0.993820,True
88,0.993060,True
89,0.991541,True
90,0.993301,True
91,0.992374,True
92,0.992398,True
93,0.992640,True
94,0.987304,True
95,0.990990,True
96,0.992891,True
97,0.992736,True
98,0.992425,True
99,0.990237,True
100,0.993595,True
101,0.993413,True
102,0.994287,True
103,0.994395,True
104,0.991902,True
105,0.993253,True
106,0.992409,True
107,0.992461,True
108,0.993303,True
109,0.992659,True
110,0.993907,True
111,0.994023,True
112,0.993061,True
113,0.993759,True
114,0.993070,True
115,0.994286,True
116,0.993206,True
117,0.993419,True
118,0.993580,True
119,0.992352,True
120,0.993332,True
121,0.994166,True
122,0.993872,True
123,0.994231,True
124,0.992070,True
125,0.993614,True
126,0.993679,True
127,0.993391,True
128,0.993040,True
129,0.991136,True
130,0.992407,True
131,0.992565,True
132,0.990685,True
133,0.991205,True
134,0.988257,True
135,0.993442,True
136,0.991730,True
137,0.993747,True
138,0.992359,True
139,0.989443,True
140,0.992388,True
141,0.993056,True
142,0.993020,True
143,0.992864,True
144,0.989375,True
145,0.993147,True
146,0.993291,True
147,0.993190,True
148,0.992478,True
149,0.982218,True
150,0.989856,True
151,0.992402,True
152,0.992482,True
153,0.993334,True
154,0.991857,True
155,0.993870,True
156,0.993533,True
157,0.994947,True
158,0.994823,True
159,0.991656,True
160,0.994320,True
161,0.994366,True
162,0.994368,True
163,0.993908,True
164,0.992286,True
165,0.994618,True
166,0.994558,True
167,0.994414,True
168,0.994630,True
169,0.992864,True
170,0.993131,True
171,0.995008,True
172,0.994176,True
173,0.994169,True
174,0.992606,True
175,0.993902,True
176,0.994117,True
177,0.994717,True
178,0.993577,True
179,0.992225,True
180,0.993697,True
181,0.994098,True
182,0.994329,True
183,0.994414,True
184,0.992388,True
185,0.993923,True
186,0.994142,True
187,0.993900,True
188,0.994227,True
189,0.992675,True
190,0.993833,True
191,0.993439,True"""

from io import StringIO
df = pd.read_csv(StringIO(data))

# Fix the Is_Train_Frame column based on the 4_5_5 split logic
# valid_train_length = 4, total_train_length = 9 (4+5), total_data_length = 14 (4+5+5)
# Train: (frame_idx % 14) < 9  --> frames 0-8, 14-22, 28-36, etc.
# Val:   (frame_idx % 14) >= 9 --> frames 9-13, 23-27, 37-41, etc.
def is_train_frame_4_5_5(frame_idx):
    valid_train_length = 4
    total_train_length = 9  # 4 + 5
    total_data_length = 14  # 4 + 5 + 5
    return (frame_idx % total_data_length) < total_train_length

# Update the column
df['Is_Train_Frame'] = df['Frame_Index'].apply(is_train_frame_4_5_5)

# Save corrected CSV
df.to_csv('hnerv_clip_similarity_corrected.csv', index=False)
print("Corrected CSV saved to: hnerv_clip_similarity_corrected.csv")

# Print summary statistics
train_frames = df[df['Is_Train_Frame'] == True]
val_frames = df[df['Is_Train_Frame'] == False]

print(f"\n=== Summary Statistics (HNeRV - Image-based) ===")
print(f"Split: 4_5_5")
print(f"Total frames: {len(df)}")
print(f"Train frames: {len(train_frames)}")
print(f"Val frames: {len(val_frames)}")
print(f"\nTrain indices (first 30): {list(train_frames['Frame_Index'].values[:30])}")
print(f"Val indices (first 10): {list(val_frames['Frame_Index'].values[:10])}")
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

ax1.scatter(train_indices, train_sims, c='green', label='Train', alpha=0.7, s=30)
ax1.scatter(val_indices, val_sims, c='orange', label='Validation', alpha=0.7, s=50, marker='s')
ax1.axhline(y=train_frames['CLIP_Similarity'].mean(), color='green', linestyle='--', 
            label=f'Train Mean: {train_frames["CLIP_Similarity"].mean():.4f}', alpha=0.5)
ax1.axhline(y=val_frames['CLIP_Similarity'].mean(), color='orange', linestyle='--', 
            label=f'Val Mean: {val_frames["CLIP_Similarity"].mean():.4f}', alpha=0.5)
ax1.set_xlabel('Frame Index', fontsize=12)
ax1.set_ylabel('CLIP Similarity', fontsize=12)
ax1.set_title('HNeRV (Image-based): CLIP Similarity - Train vs Validation (4_5_5 split)', fontsize=14, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.975, 0.996])

# Plot 2: Distribution histogram
ax2 = axes[1]
bins = np.linspace(0.975, 0.996, 30)
ax2.hist(train_sims, bins=bins, alpha=0.6, label='Train', color='green', edgecolor='black')
ax2.hist(val_sims, bins=bins, alpha=0.6, label='Validation', color='orange', edgecolor='black')
ax2.axvline(x=train_frames['CLIP_Similarity'].mean(), color='green', linestyle='--', linewidth=2, 
            label=f'Train Mean: {train_frames["CLIP_Similarity"].mean():.4f}')
ax2.axvline(x=val_frames['CLIP_Similarity'].mean(), color='orange', linestyle='--', linewidth=2, 
            label=f'Val Mean: {val_frames["CLIP_Similarity"].mean():.4f}')
ax2.set_xlabel('CLIP Similarity', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Distribution of CLIP Similarity Scores', fontsize=14, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hnerv_clip_similarity_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: hnerv_clip_similarity_analysis.png")

# Print frames with lowest CLIP similarity
print(f"\n=== Lowest CLIP Similarity Frames ===")
worst_frames = df.nsmallest(10, 'CLIP_Similarity')[['Frame_Index', 'CLIP_Similarity', 'Is_Train_Frame']]
print(worst_frames.to_string(index=False))

plt.show()
