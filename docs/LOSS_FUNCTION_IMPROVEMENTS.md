# DeepVoid Loss Function Improvements - Summary

## Problem Identified
Your model was heavily biased toward predicting "Wall" instead of "Void" due to:
- Extremely high `void_penalty=8.0` in SCCE_Class_Penalty
- High `minority_boost=3.0` making wall an attractive default
- No penalty for wall over-prediction

## Changes Made

### 1. Updated NETS_LITE.py
Added new improved loss functions in a dedicated section:

- **SCCE_Class_Penalty_Fixed**: Balanced version with reduced penalties
  - void_penalty: 2.0 (was 8.0)
  - wall_penalty: 1.0 (new)
  - minority_boost: 2.0 (was 3.0)

- **SCCE_Proportion_Aware**: Encourages matching target class proportions
  - Uses your actual data distribution [0.65, 0.25, 0.08, 0.02]
  - Penalizes deviation from expected proportions

### 2. Updated curricular.py
- Added new loss function choices: SCCE_Class_Penalty_Fixed, SCCE_Proportion_Aware
- Reduced parameters in existing SCCE_Class_Penalty
- Added proper custom objects for model serialization

### 3. Updated curricular_pred.py
- Added new loss functions to CUSTOM_OBJECTS for model loading compatibility

### 4. Removed separate file
- Consolidated all loss functions in NETS_LITE.py as requested
- Removed improved_loss_functions.py

## Recommended Next Steps

### Option 1: Use the fixed class penalty (recommended)
```bash
python curricular.py /content/drive/MyDrive/ 4 16 SCCE_Class_Penalty_Fixed \
    --BATCH_SIZE 8 --USE_ATTENTION --LAMBDA_CONDITIONING --L_VAL 10 \
    --VALIDATION_STRATEGY gradual --PREPROCESSING robust --WARMUP_EPOCHS 5
```

### Option 2: Use proportion-aware loss
```bash
python curricular.py /content/drive/MyDrive/ 4 16 SCCE_Proportion_Aware \
    --BATCH_SIZE 8 --USE_ATTENTION --LAMBDA_CONDITIONING --L_VAL 10 \
    --VALIDATION_STRATEGY target --PREPROCESSING standard
```

### Option 3: Use standard SCCE (simplest)
```bash
python curricular.py /content/drive/MyDrive/ 4 16 SCCE \
    --BATCH_SIZE 8 --USE_ATTENTION --LAMBDA_CONDITIONING --L_VAL 10 \
    --PREPROCESSING clip_extreme --WARMUP_EPOCHS 15
```

## Expected Improvements
- Model should properly predict voids as the dominant class (65.1%)
- Reduced wall over-prediction bias
- Better minority class (filament/halo) detection
- More balanced confusion matrix

The loss function improvements are now integrated into your existing codebase and ready for training!

## New Features in 2025 Update

In addition to the improved loss functions, the following new features have been added:

### Advanced Validation Strategies
- **Gradual Validation**: Progressive validation complexity increase
- **Hybrid Validation**: Monitor both target and stage performance
- **Stage Validation**: Dynamic validation matching training stage

### Advanced Preprocessing Methods
- **Robust**: Outlier clipping with median centering and scaling
- **Log Transform**: Log10 transformation with standardization
- **Clip Extreme**: Conservative outlier clipping with standardization

### Memory Optimization
- **Overlapping Subcubes**: Configurable data augmentation with spatial overlaps
- **Memory-Efficient Prediction**: Smaller batch sizes and streaming for large datasets

### Training Stability Features
- **Learning Rate Warmup**: Gradual learning rate increase to prevent early instability
- **TEST_MODE**: Quick testing with reduced data for debugging

### Example with All New Features
```bash
python curricular.py /content/drive/MyDrive/ 4 16 SCCE_Class_Penalty_Fixed \
    --USE_ATTENTION --LAMBDA_CONDITIONING --BATCH_SIZE 8 \
    --VALIDATION_STRATEGY gradual --PREPROCESSING robust \
    --WARMUP_EPOCHS 10 --L_VAL 10
```

This comprehensive update addresses the major issues identified and provides much more flexibility for different training scenarios.
