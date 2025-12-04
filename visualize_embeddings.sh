#!/bin/bash

# Visualize CLIP Embedding Space: Model vs Ground Truth
# This creates a publication-quality figure for presentations

CHECKPOINT="output/output/Kitchen_patch_comparable/Kitchen/9_10_10_pe_1.25_80_FC9_16_KS0_3_3_RED1.5_low12_blk1_1_e300_b8_lr0.001_cosine_0.1_1_0.1_Fusion6_CLIP0.2_warmup50_DEC_pshuffel_5,2,2_gelu/epoch300.pth"
DATA_PATH="data/Kitchen"
CROP_LIST="2 4"
DATA_SPLIT="9_10_10"
METHOD="tsne"  # or "umap"
OUTPUT="embedding_space_comparison.png"

# First install required packages if needed
echo "Installing required packages..."
pip install scikit-learn umap-learn

echo "Creating embedding space visualization..."
python visualize_embedding_space.py \
    --checkpoint "$CHECKPOINT" \
    --data_path "$DATA_PATH" \
    --crop_list $CROP_LIST \
    --data_split "$DATA_SPLIT" \
    --method "$METHOD" \
    --output "$OUTPUT"

echo ""
echo "Done! Check $OUTPUT and ${OUTPUT%.png}.pdf"
