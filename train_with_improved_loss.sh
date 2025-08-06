#!/bin/bash
: <<'END_COMMENT'
Example script demonstrating the new improved loss functions for DeepVoid training.

This script showcases the recommended loss functions that fix void/wall prediction bias:
1. SCCE_Class_Penalty_Fixed - Most recommended for balanced training
2. SCCE_Proportion_Aware - Maintains target class proportions  
3. SCCE_Balanced_Class_Penalty - Alternative balanced approach

These loss functions address the model bias where it predicts wall instead of void heavily.
END_COMMENT

ROOT_DIR="/content/drive/MyDrive/"
SIM="TNG"
L=0.33  # For TNG full DM
D=4     # Depth
F=16    # Filters
GRID=512  # For TNG

# RECOMMENDED: Use the fixed class penalty loss
LOSS="SCCE_Class_Penalty_Fixed"

echo "=== Training with Improved Loss Function ==="
echo "Simulation: $SIM"
echo "Lambda: $L"
echo "Depth: $D" 
echo "Filters: $F"
echo "Loss: $LOSS (RECOMMENDED - fixes void/wall bias)"
echo "Grid: $GRID"

# Enhanced training parameters for better results
BATCH_SIZE=8
EPOCHS=500
LEARNING_RATE=0.001
ATTENTION_UNET=1  # Use attention mechanism
LAMBDA_CONDITIONING=1  # Use lambda conditioning
BATCHNORM=1

python DV_MULTI_TRAIN.py "$ROOT_DIR" "$SIM" "$L" "$D" "$F" "$LOSS" "$GRID" \
    --BATCH_SIZE $BATCH_SIZE \
    --EPOCHS $EPOCHS \
    --LEARNING_RATE $LEARNING_RATE \
    --ATTENTION_UNET \
    --LAMBDA_CONDITIONING \
    --BATCHNORM \
    --MODEL_NAME_SUFFIX "_improved_loss"

echo "Training completed with improved loss function!"
echo "This model should show better void/wall prediction balance."
