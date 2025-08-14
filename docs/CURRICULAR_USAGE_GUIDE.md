# DeepVoid Curricular Training - Usage Guide

## Overview

This guide covers the usage of `curricular.py` for training and `curricular_pred.py` for prediction with the DeepVoid cosmic void detection models.

## Table of Contents

1. [Key Features Summary](#key-features-summary)
2. [Curricular Training (`curricular.py`)](#curricular-training)
3. [Validation Strategies](#validation-strategies)
4. [Preprocessing Methods](#preprocessing-methods)
5. [RSD-Preserving Rotations](#rsd-preserving-rotations)
6. [Memory Optimization](#memory-optimization)
7. [Prediction (`curricular_pred.py`)](#prediction)
8. [Loss Functions Guide](#loss-functions-guide)
9. [Common Usage Examples](#common-usage-examples)
10. [File Output Strategy](#file-output-strategy)
11. [Troubleshooting](#troubleshooting)
12. [Log Transform Preprocessing Explained](#log-transform-preprocessing-explained)

---

## Key Features Summary

The DeepVoid curricular training system includes these advanced features:

### Core Features
- **Multi-scale Curricular Learning**: Progressive training from 0.33 to 10 Mpc/h
- **Attention U-Net Architecture**: Enhanced feature extraction with attention gates
- **Lambda Conditioning**: Scale-aware training that adapts to different interparticle separations

### Advanced Validation (4 Strategies)
- **Target**: Validate on final goal (L=10) throughout training
- **Stage**: Validate on current training stage (dynamic)
- **Hybrid**: Monitor both target and stage performance
- **Gradual**: Progressive validation complexity matching curriculum progression

### Data Processing (4 Methods)
- **Standard**: Min-max normalization to [0,1] (default)
- **Robust**: Outlier clipping + median centering + standard scaling
- **Log Transform**: Log10 transformation + standardization (for extreme distributions)
- **Clip Extreme**: Conservative outlier clipping + standardization

### Training Stability
- **Learning Rate Warmup**: Gradual LR increase (0-20 epochs) for stable initialization
- **Memory Optimization**: Configurable overlapping subcubes and rotations
- **Improved Loss Functions**: Fixed void/wall prediction bias

### Additional Features
- **Redshift Space Distortions**: Realistic observational effects
- **RSD-Preserving Rotations**: Optional data augmentation that preserves line-of-sight anisotropy
- **Extra Inputs**: Galaxy colors (g-r) and flux density
- **TEST_MODE**: Quick testing with reduced datasets
- **Comprehensive Logging**: Detailed training metrics and validation tracking

---

## Curricular Training

### Basic Syntax

```bash
python curricular.py ROOT_DIR DEPTH FILTERS LOSS [OPTIONS]
```

### Required Arguments

- `ROOT_DIR`: Root directory for the project (e.g., `/ifs/groups/vogeleyGrp/`)
- `DEPTH`: Depth of the U-Net model (e.g., `4`)
- `FILTERS`: Number of initial filters (e.g., `16`)
- `LOSS`: Loss function to use (see [Loss Functions Guide](#loss-functions-guide))

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--UNIFORM_FLAG` | False | Use uniform mass subhalos for training data |
| `--BATCH_SIZE` | 8 | Batch size for training |
| `--LEARNING_RATE` | 1e-4 | Learning rate for the optimizer |
| `--LEARNING_RATE_PATIENCE` | 10 | Patience for learning rate reduction |
| `--EARLY_STOP_PATIENCE` | 10 | Patience for early stopping |
| `--L_VAL` | "10" | Interparticle separation for validation dataset |
| `--VALIDATION_STRATEGY` | "target" | Validation strategy: 'target', 'stage', 'hybrid', or 'gradual' |
| `--TARGET_LAMBDA` | None | Target lambda for validation (defaults to L_VAL) |
| `--USE_ATTENTION` | False | Use attention U-Net architecture |
| `--LAMBDA_CONDITIONING` | False | Use lambda conditioning in the model |
| `--N_EPOCHS_PER_INTER_SEP` | 50 | Epochs to train for each interparticle separation |
| `--EXTRA_INPUTS` | None | Additional inputs ('g-r' or 'r_flux_density') |
| `--ADD_RSD` | False | Add Redshift Space Distortion to inputs |
| `--PREPROCESSING` | "standard" | Preprocessing method: 'standard', 'robust', 'log_transform', 'clip_extreme' |
| `--WARMUP_EPOCHS` | 0 | Number of warmup epochs with gradual learning rate increase |
| `--NO_OVERLAPPING_SUBCUBES` | False | Disable overlapping subcubes to save memory |
| `--RSD_PRESERVING_ROTATIONS` | False | Use RSD-preserving rotations (z-axis only + xy-flips) instead of full 3D rotations |
| `--FOCAL_ALPHA` | [0.6, 0.3, 0.09, 0.02] | Alpha values for focal loss [void, wall, filament, halo] |
| `--FOCAL_GAMMA` | 1.5 | Gamma value for focal loss |

### Training Process

Curricular training progressively trains on different interparticle separations:

1. Starts with lowest separation (0.33 Mpc/h)
2. Progressively increases: 0.33 → 3 → 5 → 7 → 10 Mpc/h
3. Trains for `N_EPOCHS_PER_INTER_SEP` epochs at each level
4. Saves checkpoints at each stage

### Validation Strategies

DeepVoid supports four different validation strategies during curricular training:

#### 1. Target-based Validation (Default)
- **Purpose**: Evaluates model performance on the final goal (typically L=10 Mpc/h)
- **Usage**: `--VALIDATION_STRATEGY target --TARGET_LAMBDA 10`
- **Best for**: Assessing how well the model will perform on the final interparticle separation
- **Behavior**: Uses a fixed validation dataset throughout all training stages

#### 2. Stage-based Validation
- **Purpose**: Evaluates model performance on the current training stage
- **Usage**: `--VALIDATION_STRATEGY stage`
- **Best for**: Understanding model performance at each curricular level
- **Behavior**: Validation dataset changes with each training stage (0.33 → 3 → 5 → 7 → 10)

#### 3. Hybrid Validation
- **Purpose**: Combines both target-based and stage-based validation
- **Usage**: `--VALIDATION_STRATEGY hybrid --TARGET_LAMBDA 10`
- **Best for**: Comprehensive monitoring of both current and final performance
- **Behavior**: Tracks metrics for both target dataset and current stage dataset

#### 4. Gradual Validation
- **Purpose**: Progressive validation complexity that gradually increases validation difficulty
- **Usage**: `--VALIDATION_STRATEGY gradual`
- **Best for**: Smoother performance transitions and reduced validation shock
- **Behavior**: Intelligent mapping between training and validation stages:
  - Training L=0.33 → Validation L=0.33 (base density on base density)
  - Training L=3 → Validation L=5 (first subhalo stage on intermediate complexity)
  - Training L=5 → Validation L=5 (intermediate stage on itself)
  - Training L=7 → Validation L=10 (higher stages on final complexity)
  - Training L=10 → Validation L=10 (final stage on final complexity)

### Validation Strategy Examples

```bash
# Target-based (default): validate on L=10 throughout training
python curricular.py /ifs/groups/vogeleyGrp/ 4 16 SCCE_Class_Penalty_Fixed --VALIDATION_STRATEGY target --TARGET_LAMBDA 10

# Stage-based: validate on current training stage
python curricular.py /ifs/groups/vogeleyGrp/ 4 16 SCCE_Class_Penalty_Fixed --VALIDATION_STRATEGY stage

# Hybrid: track both target and stage performance
python curricular.py /ifs/groups/vogeleyGrp/ 4 16 SCCE_Class_Penalty_Fixed --VALIDATION_STRATEGY hybrid --TARGET_LAMBDA 10

# Gradual: progressive validation complexity
python curricular.py /ifs/groups/vogeleyGrp/ 4 16 SCCE_Class_Penalty_Fixed --VALIDATION_STRATEGY gradual
```

### Preprocessing Options

DeepVoid supports different preprocessing methods for density data to improve training stability:

#### 1. Standard (Default)
- **Usage**: `--PREPROCESSING standard`
- **Method**: Min-max scaling to [0,1] range
- **Best for**: General purpose, stable training

#### 2. Robust
- **Usage**: `--PREPROCESSING robust`
- **Method**: Clips outliers (1st-99th percentile), median centering, std scaling, caps extreme values to [-3,3]
- **Best for**: Data with outliers causing training instability

#### 3. Log Transform
- **Usage**: `--PREPROCESSING log_transform`
- **Method**: Log10 transform followed by standardization
- **Best for**: Density fields with wide dynamic range (see detailed explanation below)

#### 4. Clip Extreme
- **Usage**: `--PREPROCESSING clip_extreme`
- **Method**: Clips extreme outliers (0.1st-99.9th percentile) then standardizes
- **Best for**: Conservative outlier handling

### Learning Rate Warmup

Learning rate warmup gradually increases the learning rate from a small value to the target value over the first few epochs:

#### Usage
```bash
# Add 10 epochs of warmup (only applied to first interparticle separation)
python curricular.py /ifs/groups/vogeleyGrp/ 4 16 SCCE --WARMUP_EPOCHS 10 --LEARNING_RATE 1e-4
```

#### Benefits
- Prevents early training instability
- Helps with convergence when using large learning rates
- Particularly useful with complex loss functions

---

### RSD-Preserving Rotations

When working with data that contains Redshift-Space Distortions (RSD), standard 3D rotations can destroy the anisotropic line-of-sight information that RSD modeling requires. DeepVoid provides RSD-preserving rotations as an alternative data augmentation strategy.

#### Automatic RSD-Preserving Mode

When the `--ADD_RSD` flag is used, RSD-preserving rotations are **automatically enabled** as the default augmentation strategy. This ensures that the line-of-sight anisotropy crucial for RSD analysis is preserved during training.

#### Manual Control

You can also manually enable RSD-preserving rotations for any dataset using:
```bash
--RSD_PRESERVING_ROTATIONS
```

#### Memory Optimization

To balance data augmentation benefits with GPU memory constraints, DeepVoid uses lighter augmentation levels by default:

- **Default (Light) Augmentation:**
  - RSD-preserving: 4x samples per subcube (4 z-axis rotations only)
  - Standard: 2x samples per subcube (original + one rotation)

- **Extra (Heavy) Augmentation:** Use `--EXTRA_AUGMENTATION` for maximum augmentation:
  - RSD-preserving: 8x samples per subcube (4 z-rotations + 4 z-rotations with xy-flips)
  - Standard: 4x samples per subcube (full 3D rotations)

#### Technical Details

**RSD-Preserving Augmentation Strategy:**
- Rotations only around the z-axis (line-of-sight direction)
- Preserves RSD anisotropy while still providing data augmentation
- Light version: 0°, 90°, 180°, 270° z-axis rotations
- Heavy version: Above rotations + xy-flipped versions of each

**Comparison with Standard Rotations:**
- Standard rotations: Full 3D rotations around x, y, z axes
- RSD-preserving: Only z-axis rotations + optional xy-flips
- Memory usage: Light versions use ~50% less memory than heavy versions

#### Usage Examples

**Automatic RSD-preserving with light augmentation (recommended for RSD data):**
```bash
python curricular.py /path/to/density.dat /path/to/mask.dat --ADD_RSD --subgrid_dim 32
# Automatically enables RSD-preserving rotations with 4x augmentation
```

**Manual RSD-preserving with heavy augmentation:**
```bash
python curricular.py /path/to/density.dat /path/to/mask.dat --RSD_PRESERVING_ROTATIONS --EXTRA_AUGMENTATION --subgrid_dim 32
# Uses RSD-preserving rotations with 8x augmentation
```

**Memory-optimized training for large datasets:**
```bash
python curricular.py /path/to/density.dat /path/to/mask.dat --subgrid_dim 32
# Uses light standard augmentation (2x) for minimal memory usage
```

**Maximum augmentation for small datasets:**
```bash
python curricular.py /path/to/density.dat /path/to/mask.dat --EXTRA_AUGMENTATION --subgrid_dim 32
# Uses heavy standard augmentation (4x) for maximum data diversity
```

---

## Memory Optimization

### Overlapping Subcubes with Rotations

By default, curricular training uses overlapping subcubes with rotations for better data augmentation. This can be disabled to save memory:

#### Usage
```bash
# Disable overlapping subcubes to save memory (may reduce model performance)
python curricular.py /ifs/groups/vogeleyGrp/ 4 16 SCCE --NO_OVERLAPPING_SUBCUBES
```

#### Benefits of Default (Overlapping) Approach:
- Better data augmentation through spatial overlaps
- Improved model generalization
- More robust training with limited data

#### Benefits of Disabled Approach:
- Reduced memory usage (useful for large models or limited GPU memory)
- Faster data loading
- Simpler data pipeline

---

## Prediction

### Basic Syntax

```bash
python curricular_pred.py ROOT_DIR MODEL_NAME L_PRED [OPTIONS]
```

### Required Arguments

- `ROOT_DIR`: Root directory for the project (e.g., `/ifs/groups/vogeleyGrp/`)
- `MODEL_NAME`: Name of the trained model (without file extension)
- `L_PRED`: Interparticle separation for prediction (e.g., "10")

### Key Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--BATCH_SIZE` | 8 | Batch size for prediction |
| `--MAX_PRED_BATCHES` | None | Limit number of batches (for memory) |
| `--SAVE_PREDICTIONS` | True | Save prediction arrays |
| `--SKIP_SLICE_PLOTS` | False | Skip generating visualization plots |
| `--TEST_MODE` | False | Use only 16 samples for quick testing |
| `--PREPROCESSING` | "standard" | Preprocessing method: 'standard', 'robust', 'log_transform', 'clip_extreme' |

### Important Note on Preprocessing

**WARNING: The preprocessing method used for prediction MUST match the preprocessing method used during training**

Models trained with different preprocessing methods need to be evaluated with their respective preprocessing. Check your training logs or configuration to determine which preprocessing was used.

### Prediction Examples

```bash
# Standard preprocessing (default)
python curricular_pred.py /ifs/groups/vogeleyGrp/ TNG_curricular_model 10 --PREPROCESSING standard

# Robust preprocessing
python curricular_pred.py /ifs/groups/vogeleyGrp/ TNG_curricular_model 10 --PREPROCESSING robust

# Log transform preprocessing
python curricular_pred.py /ifs/groups/vogeleyGrp/ TNG_curricular_model 10 --PREPROCESSING log_transform

# Memory-safe prediction
python curricular_pred.py /ifs/groups/vogeleyGrp/ MODEL_NAME 10 --MAX_PRED_BATCHES 32 --BATCH_SIZE 4

# Quick test prediction
python curricular_pred.py /ifs/groups/vogeleyGrp/ MODEL_NAME 10 --TEST_MODE --SKIP_SLICE_PLOTS
```

---

## File Output Strategy

### Prediction File Naming

The `curricular_pred.py` script ensures that all output files include both the L_PRED (interparticle separation) value and the preprocessing method in their filenames. This prevents file overwrites when running predictions multiple times with different parameters.

#### File Naming Patterns

**Prediction Files (PRED_PATH)**
- Main predictions: `{MODEL_NAME}_predictions_L{L_PRED}_{PREPROCESSING}.npy`
- Slice predictions: `{MODEL_NAME}_slice_predictions_L{L_PRED}_{PREPROCESSING}.npy`
- Legacy .fvol format: `{MODEL_NAME}_predictions_L{L_PRED}_{PREPROCESSING}.fvol`

**Figure Files (FIG_PATH/{MODEL_NAME}/)**
- Confusion matrix: `{MODEL_NAME}_confusion_matrix_L{L_PRED}_{PREPROCESSING}.npy`
- Slice plot data: `{MODEL_NAME}_slice_data_L{L_PRED}_{PREPROCESSING}.npz`
- ROC curves: `{MODEL_NAME}_temp_L{L_PRED}_{PREPROCESSING}_ROC_OvR.png`
- PR curves: `{MODEL_NAME}_temp_L{L_PRED}_{PREPROCESSING}_PR.png`
- Confusion matrix plot: `{MODEL_NAME}_temp_L{L_PRED}_{PREPROCESSING}_cm.png`
- Prediction comparison: `{MODEL_NAME}_temp_L{L_PRED}_{PREPROCESSING}-pred-comp.png`

#### Example Filenames

For model "TNG_curricular_model" with L_PRED=10 and PREPROCESSING=robust:

```
TNG_curricular_model_predictions_L10_robust.npy
TNG_curricular_model_confusion_matrix_L10_robust.npy
TNG_curricular_model_temp_L10_robust_ROC_OvR.png
TNG_curricular_model_temp_L10_robust-pred-comp.png
```

### Training Process

Curricular training progressively trains on different interparticle separations:

1. Starts with lowest separation (0.33 Mpc/h)
2. Progressively increases: 0.33 → 3 → 5 → 7 → 10 Mpc/h
3. Trains for `N_EPOCHS_PER_INTER_SEP` epochs at each level
4. Saves checkpoints at each stage

### Validation Strategies

DeepVoid supports three different validation strategies during curricular training:

#### 1. Target-based Validation (Default)
- **Purpose**: Evaluates model performance on the final goal (typically L=10 Mpc/h)
- **Usage**: `--VALIDATION_STRATEGY target --TARGET_LAMBDA 10`
- **Best for**: Assessing how well the model will perform on the final interparticle separation
- **Behavior**: Uses a fixed validation dataset throughout all training stages

#### 2. Stage-based Validation
- **Purpose**: Evaluates model performance on the current training stage
- **Usage**: `--VALIDATION_STRATEGY stage`
- **Best for**: Understanding model performance at each curricular level
- **Behavior**: Validation dataset changes with each training stage (0.33 → 3 → 5 → 7 → 10)

#### 3. Hybrid Validation
- **Purpose**: Combines both target-based and stage-based validation
- **Usage**: `--VALIDATION_STRATEGY hybrid --TARGET_LAMBDA 10`
- **Best for**: Comprehensive monitoring of both current and final performance
- **Behavior**: Tracks metrics for both target dataset and current stage dataset

### Validation Strategy Examples

```bash
# Target-based (default): validate on L=10 throughout training
python curricular.py /path/to/data/ 4 16 SCCE_Class_Penalty_Fixed --VALIDATION_STRATEGY target --TARGET_LAMBDA 10

# Stage-based: validate on current training stage
python curricular.py /path/to/data/ 4 16 SCCE_Class_Penalty_Fixed --VALIDATION_STRATEGY stage

# Hybrid: track both target and stage performance
python curricular.py /path/to/data/ 4 16 SCCE_Class_Penalty_Fixed --VALIDATION_STRATEGY hybrid --TARGET_LAMBDA 10
```

### Preprocessing Options

DeepVoid supports different preprocessing methods for density data to improve training stability:

#### 1. Standard (Default)
- **Usage**: `--PREPROCESSING standard`
- **Method**: Min-max scaling to [0,1] range
- **Best for**: General purpose, stable training

#### 2. Robust
- **Usage**: `--PREPROCESSING robust`
- **Method**: Clips outliers (1st-99th percentile), median centering, std scaling, caps extreme values to [-3,3]
- **Best for**: Data with outliers causing training instability

#### 3. Log Transform
- **Usage**: `--PREPROCESSING log_transform`
- **Method**: Log10 transform followed by standardization
- **Best for**: Density fields with wide dynamic range

#### 4. Clip Extreme
- **Usage**: `--PREPROCESSING clip_extreme`
- **Method**: Clips extreme outliers (0.1st-99.9th percentile) then standardizes
- **Best for**: Conservative outlier handling

### Learning Rate Warmup

Learning rate warmup gradually increases the learning rate from a small value to the target value over the first few epochs:

#### Usage
```bash
# Add 10 epochs of warmup (only applied to first interparticle separation)
python curricular.py /path/to/data/ 4 16 SCCE --WARMUP_EPOCHS 10 --LEARNING_RATE 1e-4
```

#### Benefits
- Prevents early training instability
- Helps with convergence when using large learning rates
- Particularly useful with complex loss functions

---

## Prediction

### Basic Syntax

```bash
python curricular_pred.py MODEL_NAME L_PRED [OPTIONS]
```

### Required Arguments

- `MODEL_NAME`: Name of the trained model (without file extension)
- `L_PRED`: Interparticle separation for prediction (e.g., "10")

### Key Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--ROOT_DIR` | "/content/drive/MyDrive/" | Root directory path |
| `--BATCH_SIZE` | 8 | Batch size for prediction |
| `--MAX_PRED_BATCHES` | None | Limit number of batches (for memory) |
| `--SAVE_PREDICTIONS` | True | Save prediction arrays |
| `--SKIP_SLICE_PLOTS` | False | Skip generating visualization plots |
| `--TEST_MODE` | False | Use only 16 samples for quick testing |

---

## Loss Functions Guide

### Recommended Loss Functions

#### 1. `SCCE_Class_Penalty_Fixed` **(RECOMMENDED)**
- **Best for**: Balanced void detection without bias
- **Features**: Reduced penalties, balanced wall/void treatment
- **Parameters**: `void_penalty=2.0`, `wall_penalty=1.0`, `minority_boost=2.0`

#### 2. `SCCE_Proportion_Aware`
- **Best for**: Maintaining exact class proportions
- **Features**: Penalizes deviation from target proportions [65%, 25%, 8%, 2%]
- **Parameters**: `target_props=[0.65, 0.25, 0.08, 0.02]`, `prop_weight=1.0`

#### 3. `SCCE` 
- **Best for**: Simplest, most stable training
- **Features**: Standard sparse categorical crossentropy, no custom penalties
- **Use when**: Other loss functions cause instability

### Advanced Loss Functions

#### 4. `SCCE_Class_Penalty` (Updated)
- **Features**: Original penalty function with balanced parameters
- **Parameters**: `void_penalty=2.0` (reduced from 8.0), `minority_boost=1.5`

#### 5. `SCCE_Balanced_Class_Penalty`
- **Features**: Balanced penalties for all classes
- **Parameters**: `void_penalty=1.5`, `wall_penalty=1.5`, `minority_boost=2.0`

#### 6. `SCCE_Void_Penalty`
- **Features**: Specifically targets void over-prediction
- **Parameters**: `max_void_fraction=0.7`, `penalty_factor=5.0`

### Experimental Loss Functions

#### 7. `FOCAL_CCE`
- **Features**: Focal loss for handling class imbalance
- **Parameters**: `alpha=[0.4, 0.4, 0.15, 0.05]`, `gamma=2.0`

#### 8. `DISCCE`
- **Features**: Combines SCCE with Dice loss
- **Parameters**: `cce_weight=1.0`, `dice_weight=1.0`

---

## Common Usage Examples

### 1. Production Training (Recommended)
```bash
# Best practices with all recommended features
python curricular.py /ifs/groups/vogeleyGrp/ 4 16 SCCE_Class_Penalty_Fixed \
    --USE_ATTENTION --LAMBDA_CONDITIONING --BATCH_SIZE 8 \
    --VALIDATION_STRATEGY gradual --PREPROCESSING robust \
    --WARMUP_EPOCHS 10 --L_VAL 10
```

### 2. Fast Experimental Training
```bash
# Quick testing with reduced complexity
python curricular.py /ifs/groups/vogeleyGrp/ 3 8 SCCE \
    --BATCH_SIZE 16 --VALIDATION_STRATEGY stage \
    --N_EPOCHS_PER_INTER_SEP 20
```

### 3. Memory-Constrained Training
```bash
# For limited GPU memory
python curricular.py /ifs/groups/vogeleyGrp/ 4 16 SCCE_Class_Penalty_Fixed \
    --BATCH_SIZE 4 --NO_OVERLAPPING_SUBCUBES \
    --VALIDATION_STRATEGY target --L_VAL 10
```

### 4. Difficult Data with Extreme Values
```bash
# For data with outliers or extreme distributions
python curricular.py /ifs/groups/vogeleyGrp/ 4 16 SCCE \
    --PREPROCESSING log_transform --WARMUP_EPOCHS 20 \
    --LEARNING_RATE 1e-5 --VALIDATION_STRATEGY hybrid
```

### 5. Comprehensive Monitoring Training
```bash
# With hybrid validation and all stability features
python curricular.py /ifs/groups/vogeleyGrp/ 4 16 SCCE_Proportion_Aware \
    --USE_ATTENTION --LAMBDA_CONDITIONING --BATCH_SIZE 8 \
    --VALIDATION_STRATEGY hybrid --TARGET_LAMBDA 10 \
    --PREPROCESSING clip_extreme --WARMUP_EPOCHS 5
```

### 6. Conservative Stable Training
```bash
# Maximum stability for problematic datasets
python curricular.py /ifs/groups/vogeleyGrp/ 3 16 SCCE \
    --BATCH_SIZE 16 --LEARNING_RATE 1e-5 \
    --PREPROCESSING standard --WARMUP_EPOCHS 25 \
    --VALIDATION_STRATEGY target --L_VAL 10
```

### Basic Training (Recommended)

```bash
# Best balanced approach with robust preprocessing and warmup
python curricular.py /content/drive/MyDrive/ 4 16 SCCE_Class_Penalty_Fixed \
    --BATCH_SIZE 8 --USE_ATTENTION --LAMBDA_CONDITIONING --L_VAL 10 \
    --VALIDATION_STRATEGY target --TARGET_LAMBDA 10 \
    --PREPROCESSING robust --WARMUP_EPOCHS 10

# Proportion-aware training with log transform for wide dynamic range data
python curricular.py /content/drive/MyDrive/ 4 16 SCCE_Proportion_Aware \
    --BATCH_SIZE 8 --USE_ATTENTION --LAMBDA_CONDITIONING --L_VAL 10 \
    --VALIDATION_STRATEGY stage \
    --PREPROCESSING log_transform --WARMUP_EPOCHS 5

# Safe fallback with hybrid validation and standard preprocessing
python curricular.py /content/drive/MyDrive/ 4 16 SCCE \
    --BATCH_SIZE 8 --USE_ATTENTION --LAMBDA_CONDITIONING --L_VAL 10 \
    --VALIDATION_STRATEGY hybrid --TARGET_LAMBDA 10 \
    --PREPROCESSING standard --WARMUP_EPOCHS 0
```

### Advanced Training Options

```bash
# High-capacity model with attention
python curricular.py /content/drive/MyDrive/ 5 32 SCCE_Class_Penalty_Fixed \
    --BATCH_SIZE 4 --USE_ATTENTION --LAMBDA_CONDITIONING \
    --LEARNING_RATE 5e-5 --N_EPOCHS_PER_INTER_SEP 75

# With additional inputs (color information)
python curricular.py /content/drive/MyDrive/ 4 16 SCCE_Class_Penalty_Fixed \
    --EXTRA_INPUTS g-r --USE_ATTENTION --LAMBDA_CONDITIONING

# Memory-constrained training
python curricular.py /content/drive/MyDrive/ 3 8 SCCE \
    --BATCH_SIZE 4 --N_EPOCHS_PER_INTER_SEP 30
```

### Prediction Examples

```bash
# Standard prediction
python curricular_pred.py TNG_curricular_SCCE_Class_Penalty_Fixed_D4_F16_attention_lambda_2025-08-06_03-00-21 10

# Memory-safe prediction
python curricular_pred.py MODEL_NAME 10 --MAX_PRED_BATCHES 32 --BATCH_SIZE 4

# Quick test prediction
python curricular_pred.py MODEL_NAME 10 --TEST_MODE --SKIP_SLICE_PLOTS
```

---

## Troubleshooting

### Problem: Model predicts too many walls, not enough voids

**Solution**: Use balanced loss functions
```bash
# Try the fixed class penalty
python curricular.py /path/to/data 4 16 SCCE_Class_Penalty_Fixed --BATCH_SIZE 8

# Or use proportion-aware loss
python curricular.py /path/to/data 4 16 SCCE_Proportion_Aware --BATCH_SIZE 8
```

### Problem: Training instability or exploding gradients

**Solution**: Reduce learning rate and use gradient clipping
```bash
python curricular.py /path/to/data 4 16 SCCE \
    --LEARNING_RATE 1e-5 --BATCH_SIZE 4
```

### Problem: Out of memory errors

**Solutions**:
1. Reduce batch size: `--BATCH_SIZE 4` or `--BATCH_SIZE 2`
2. Reduce model size: Use `DEPTH=3` and `FILTERS=8`
3. For prediction: Use `--MAX_PRED_BATCHES 16`

### Problem: Poor minority class detection (filaments/halos)

**Solutions**:
1. Use `SCCE_Class_Penalty_Fixed` with higher `minority_boost`
2. Try `FOCAL_CCE` loss for better class imbalance handling
3. Increase training epochs: `--N_EPOCHS_PER_INTER_SEP 100`

### Problem: Model loading errors during prediction

**Solution**: Ensure custom loss functions are in CUSTOM_OBJECTS
- Check that your loss function is listed in `curricular_pred.py` CUSTOM_OBJECTS
- Use the exact same loss function name for training and prediction

---

## Performance Monitoring

### During Training
- Monitor void fraction with `VoidFractionMonitor` callback
- Watch for balanced accuracy and F1 scores
- Check that validation loss decreases steadily

### During Prediction
- Check confusion matrices for class balance
- Monitor precision/recall for each class
- Verify that void predictions match expected ~65% proportion

### Using the Diagnosis Tool

```bash
# Analyze training logs for bias issues
python diagnose_bias.py /path/to/training_log.txt
```

---

## File Outputs

### Training Outputs
- Model checkpoints: `MODEL_NAME_L{separation}.keras`
- Weight files: `MODEL_NAME_L{separation}_weights.h5`
- Training history: `MODEL_NAME_training_history.png`
- Hyperparameters: `MODEL_NAME_hyperparameters.txt`

### Prediction Outputs
- Predictions: `MODEL_NAME_predictions_L{separation}.npy`
- Confusion matrix: `MODEL_NAME_confusion_matrix_L{separation}.npy`
- Slice visualizations: Various `.png` files
- Metrics: Saved in model figure directory

---

## Best Practices

1. **Start Simple**: Begin with `SCCE` loss to establish baseline
2. **Use Attention**: `--USE_ATTENTION` generally improves performance
3. **Lambda Conditioning**: `--LAMBDA_CONDITIONING` helps with multi-scale training
4. **Monitor Training**: Watch void fraction and class balance during training
5. **Validate Results**: Always check confusion matrices and class-specific metrics
6. **Save Intermediate**: Keep checkpoints at each interparticle separation
7. **Memory Management**: Use appropriate batch sizes for your hardware

The curricular training approach provides robust, multi-scale void detection with careful handling of class imbalance issues!

---

## Log Transform Preprocessing Explained

### What is Log Transform Preprocessing?

Log transform preprocessing applies a logarithmic transformation to the density data before feeding it to the neural network. In DeepVoid, this specifically uses a **log₁₀ transformation** followed by standardization.

### Why Use Log Transform?

Cosmic density fields have **extreme dynamic ranges** - meaning they can have very low density values (in voids) and very high density values (in dense regions like halos). This creates several problems for neural networks:

1. **Extreme Value Ranges**: Raw density values might range from near 0 in voids to thousands in dense regions
2. **Skewed Distributions**: Most of the volume is low-density (voids and walls), with only small regions of high density
3. **Training Instability**: Neural networks struggle with such wide ranges of input values
4. **Gradient Problems**: Very large values can cause exploding gradients, while very small values lead to vanishing gradients

### How Log Transform Helps

The logarithmic transformation **compresses the dynamic range**:

- **Before**: Density values from 0.1 to 10,000 (5 orders of magnitude)
- **After log₁₀**: Values from -1 to 4 (compressed to ~5 units)

This compression makes the data:
- **More normally distributed**: The transformation makes the skewed density distribution more symmetric
- **Easier to learn**: Neural networks perform better with inputs in moderate ranges
- **More stable**: Reduces gradient explosion/vanishing problems
- **Better balanced**: Gives equal "weight" to low and high density regions in the learning process

### When to Use Log Transform

Use log transform preprocessing when:
- Your density data has very wide dynamic ranges (common in cosmological simulations)
- Standard preprocessing leads to training instability
- The model has trouble learning void structures (very low density regions)
- You notice the loss function oscillating wildly during training

### Real-World Example

In cosmic void detection:
- **Voids** might have density contrast δ = -0.9 (very underdense)
- **Walls** might have δ = 0.1 (slightly overdense) 
- **Halos** might have δ = 100+ (extremely overdense)

After log transform, these become more manageable values that the neural network can learn from more effectively, allowing it to distinguish between these different cosmic structures with better accuracy.

**Important**: Always use the same preprocessing method for both training AND prediction - if you train with log transform, you must also predict with log transform!
