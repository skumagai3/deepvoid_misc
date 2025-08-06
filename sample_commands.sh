#!/bin/bash

# Sample curricular training commands with new features

echo "=== Sample DeepVoid Curricular Training Commands ==="
echo ""

echo "1. Basic training with robust preprocessing and warmup:"
echo "python curricular.py /path/to/data/ 4 16 SCCE_Class_Penalty_Fixed \\"
echo "    --BATCH_SIZE 8 --USE_ATTENTION --LAMBDA_CONDITIONING \\"
echo "    --PREPROCESSING robust --WARMUP_EPOCHS 10 \\"
echo "    --VALIDATION_STRATEGY target --TARGET_LAMBDA 10"
echo ""

echo "2. Training for problematic data with log transform:"
echo "python curricular.py /path/to/data/ 4 16 SCCE \\"
echo "    --BATCH_SIZE 4 --LEARNING_RATE 1e-5 \\"
echo "    --PREPROCESSING log_transform --WARMUP_EPOCHS 15 \\"
echo "    --VALIDATION_STRATEGY stage"
echo ""

echo "3. Comprehensive monitoring with hybrid validation:"
echo "python curricular.py /path/to/data/ 4 16 SCCE_Proportion_Aware \\"
echo "    --BATCH_SIZE 8 --USE_ATTENTION --LAMBDA_CONDITIONING \\"
echo "    --PREPROCESSING clip_extreme --WARMUP_EPOCHS 5 \\"
echo "    --VALIDATION_STRATEGY hybrid --TARGET_LAMBDA 10 \\"
echo "    --LEARNING_RATE 5e-5"
echo ""

echo "4. Conservative approach for stable training:"
echo "python curricular.py /path/to/data/ 3 16 SCCE \\"
echo "    --BATCH_SIZE 16 --LEARNING_RATE 1e-5 \\"
echo "    --PREPROCESSING standard --WARMUP_EPOCHS 20 \\"
echo "    --VALIDATION_STRATEGY target --L_VAL 10"
echo ""

echo "=== Preprocessing Options ==="
echo "• standard: Min-max scaling [0,1] (default)"
echo "• robust: Outlier clipping + median centering + std scaling"
echo "• log_transform: Log10 transform + standardization"
echo "• clip_extreme: Conservative outlier clipping + standardization"
echo ""

echo "=== Learning Rate Warmup ==="
echo "• WARMUP_EPOCHS=0: No warmup (default)"
echo "• WARMUP_EPOCHS=5-10: Gentle warmup (recommended)"
echo "• WARMUP_EPOCHS=15-20: Extended warmup for difficult data"
echo ""

echo "=== Validation Strategies ==="
echo "• target: Validate on final goal (L=10) throughout training"
echo "• stage: Validate on current training stage (dynamic)"
echo "• hybrid: Monitor both target and stage performance"
