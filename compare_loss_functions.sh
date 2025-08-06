#!/bin/bash
: <<'END_COMMENT'
Example script for testing different improved loss functions.

This script provides easy configuration switching between the new loss functions
to help users find the best one for their specific dataset and requirements.
END_COMMENT

ROOT_DIR="/content/drive/MyDrive/"
SIM="TNG"
L=0.33
D=4
F=16
GRID=512

# Choose one of the improved loss functions:
# Uncomment the loss function you want to test

# Option 1: Most recommended - balanced approach
LOSS="SCCE_Class_Penalty_Fixed"
MODEL_SUFFIX="_fixed_penalty"

# Option 2: Maintains target proportions (65% void, 25% wall, etc.)
# LOSS="SCCE_Proportion_Aware"
# MODEL_SUFFIX="_proportion_aware"

# Option 3: Alternative balanced approach
# LOSS="SCCE_Balanced_Class_Penalty"  
# MODEL_SUFFIX="_balanced_penalty"

# Option 4: Safe fallback - standard SCCE without penalties
# LOSS="SCCE"
# MODEL_SUFFIX="_standard_scce"

echo "=== Loss Function Comparison Training ==="
echo "Selected Loss: $LOSS"
echo "Model Suffix: $MODEL_SUFFIX"

# Standard training parameters
BATCH_SIZE=8
EPOCHS=100  # Shorter for comparison runs
LEARNING_RATE=0.001

python DV_MULTI_TRAIN.py "$ROOT_DIR" "$SIM" "$L" "$D" "$F" "$LOSS" "$GRID" \
    --BATCH_SIZE $BATCH_SIZE \
    --EPOCHS $EPOCHS \
    --LEARNING_RATE $LEARNING_RATE \
    --ATTENTION_UNET \
    --MODEL_NAME_SUFFIX "$MODEL_SUFFIX"

echo "Training completed for loss function: $LOSS"
echo "Compare results between different loss functions to find the best for your data."
