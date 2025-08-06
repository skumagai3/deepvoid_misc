#!/bin/bash
: <<'END_COMMENT'
Quick fix script for existing models with void/wall prediction bias.

If you have a model that heavily predicts wall instead of void, use this script
to retrain with the improved loss functions.
END_COMMENT

# Check if user provided the problematic model name
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <MODEL_NAME>"
    echo "Example: $0 BASE_TNG_L0.33_D4_F16_LOSS_SCCE_Class_Penalty"
    echo ""
    echo "This script will retrain your model with improved loss function to fix void/wall bias."
    exit 1
fi

ORIGINAL_MODEL="$1"
ROOT_DIR="/content/drive/MyDrive/"

# Extract parameters from model name (basic parsing)
if [[ $ORIGINAL_MODEL == *"TNG"* ]]; then
    SIM="TNG"
    GRID=512
elif [[ $ORIGINAL_MODEL == *"BOL"* ]]; then
    SIM="BOL" 
    GRID=640
else
    echo "Could not determine simulation type from model name. Please check the model name."
    exit 1
fi

# Default parameters (adjust as needed)
L=0.33
D=4
F=16

# Use the improved loss function
LOSS="SCCE_Class_Penalty_Fixed"
NEW_MODEL_SUFFIX="_FIXED"

echo "=== Fixing Void/Wall Prediction Bias ==="
echo "Original Model: $ORIGINAL_MODEL"
echo "New Loss Function: $LOSS"
echo "This will create: ${ORIGINAL_MODEL}${NEW_MODEL_SUFFIX}"
echo ""

# Ask for confirmation
read -p "Continue with retraining? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Retraining cancelled."
    exit 1
fi

# Run training with improved loss
python DV_MULTI_TRAIN.py "$ROOT_DIR" "$SIM" "$L" "$D" "$F" "$LOSS" "$GRID" \
    --MODEL_NAME_SUFFIX "$NEW_MODEL_SUFFIX" \
    --BATCH_SIZE 8 \
    --EPOCHS 300 \
    --LEARNING_RATE 0.001 \
    --ATTENTION_UNET \
    --LAMBDA_CONDITIONING

echo ""
echo "=== Retraining Complete ==="
echo "Original model: $ORIGINAL_MODEL"  
echo "New improved model: ${ORIGINAL_MODEL}${NEW_MODEL_SUFFIX}"
echo ""
echo "The new model should show much better void/wall prediction balance!"
echo "Test it with the prediction script to verify the improvement."
