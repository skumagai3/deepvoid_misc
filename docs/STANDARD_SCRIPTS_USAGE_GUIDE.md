# DeepVoid Training Scripts - Usage Guide

## Overview

This guide covers the usage of the standard DeepVoid training scripts for cosmic void detection models. These scripts provide single-scale training (as opposed to the curricular multi-scale approach).

## Table of Contents

1. [Training Scripts Overview](#training-scripts-overview)
2. [DV_MULTI_TRAIN.py - Main Training Script](#dv_multi_trainpy)
3. [DV_MULTI_PRED.py - Prediction Script](#dv_multi_predpy)
4. [DV_MULTI_TRANSFER.py - Transfer Learning](#dv_multi_transferpy)
5. [Loss Functions Guide](#loss-functions-guide)
6. [Common Usage Examples](#common-usage-examples)
7. [Troubleshooting](#troubleshooting)

---

## Training Scripts Overview

### Main Scripts

- **`DV_MULTI_TRAIN.py`**: Primary training script for single-scale models
- **`DV_MULTI_PRED.py`**: Prediction and evaluation script
- **`DV_MULTI_TRANSFER.py`**: Transfer learning between different datasets
- **`attention_test.py`**: Experimental attention mechanism testing

### Key Differences from Curricular Training

| Feature | Standard Training | Curricular Training |
|---------|------------------|-------------------|
| **Scale** | Single interparticle separation | Multi-scale progression |
| **Training Time** | Faster, single stage | Longer, multiple stages |
| **Complexity** | Simpler setup | More complex curriculum |
| **Use Case** | Quick experiments, baselines | Production models |

---

## DV_MULTI_TRAIN.py

### Basic Syntax

```bash
python DV_MULTI_TRAIN.py ROOT_DIR SIM L DEPTH FILTERS LOSS GRID [OPTIONS]
```

### Required Arguments

- `ROOT_DIR`: Root directory for the project
- `SIM`: Simulation type (`TNG` or `BOL`)
- `L`: Interparticle separation in Mpc/h (e.g., `10`)
- `DEPTH`: Depth of the U-Net model (e.g., `4`)
- `FILTERS`: Number of initial filters (e.g., `16`)
- `LOSS`: Loss function (see [Loss Functions Guide](#loss-functions-guide))
- `GRID`: Grid size (e.g., `512`)

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--UNIFORM_FLAG` | False | Use identical masses for all subhaloes |
| `--BATCHNORM` | False | Use batch normalization |
| `--DROPOUT` | 0.0 | Dropout rate |
| `--MULTI_FLAG` | False | Use multiprocessing |
| `--LOW_MEM_FLAG` | True | Load less training data and report fewer metrics |
| `--FOCAL_ALPHA` | [0.25,0.25,0.25,0.25] | Focal loss alpha parameters |
| `--FOCAL_GAMMA` | 2.0 | Focal loss gamma parameter |
| `--BATCH_SIZE` | 16 | Batch size for training |
| `--EPOCHS` | 150 | Maximum number of epochs |
| `--LEARNING_RATE` | 1e-4 | Learning rate |
| `--LEARNING_RATE_PATIENCE` | 10 | Patience for learning rate reduction |
| `--PATIENCE` | 25 | Patience for early stopping |
| `--REGULARIZE_FLAG` | False | Use L2 regularization |
| `--TENSORBOARD_FLAG` | False | Use TensorBoard logging |
| `--BINARY_MASK` | False | Use binary mask (void vs non-void) |
| `--BOUNDARY_MASK` | None | Boundary mask for loss calculation |
| `--EXTRA_INPUTS` | None | Additional inputs ('g-r' or 'r_flux_density') |
| `--ADD_RSD` | False | Add Redshift Space Distortion |
| `--USE_PCONV` | False | Use partial convolutions |
| `--ATTENTION_UNET` | False | Use attention U-Net architecture |
| `--LAMBDA_CONDITIONING` | False | Use lambda conditioning |

### Example Usage

```bash
# Basic training
python DV_MULTI_TRAIN.py /content/drive/MyDrive/ TNG 10 4 16 SCCE_Class_Penalty_Fixed 512 \
    --BATCH_SIZE 8 --ATTENTION_UNET --LAMBDA_CONDITIONING

# Advanced training with attention and extra inputs
python DV_MULTI_TRAIN.py /content/drive/MyDrive/ TNG 10 4 32 SCCE_Proportion_Aware 512 \
    --BATCH_SIZE 4 --ATTENTION_UNET --BATCHNORM --DROPOUT 0.1 \
    --EXTRA_INPUTS g-r --LEARNING_RATE 5e-5
```

---

## DV_MULTI_PRED.py

### Basic Syntax

```bash
python DV_MULTI_PRED.py ROOT_DIR SIM MODEL_NAME L [OPTIONS]
```

### Required Arguments

- `ROOT_DIR`: Root directory for the project
- `SIM`: Simulation type (`TNG` or `BOL`)
- `MODEL_NAME`: Name of the trained model
- `L`: Interparticle separation for prediction

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--GRID` | 512 | Grid size for prediction |
| `--BATCH_SIZE` | 16 | Batch size for prediction |
| `--XOVER_FLAG` | False | Cross-over prediction flag |

### Example Usage

```bash
# Standard prediction
python DV_MULTI_PRED.py /content/drive/MyDrive/ TNG TNG-D4-F16-Nm512-th0.65-sig2.4-L10-SCCE_Class_Penalty_Fixed 10

# Prediction with different grid size
python DV_MULTI_PRED.py /content/drive/MyDrive/ TNG MODEL_NAME 10 --GRID 256 --BATCH_SIZE 8
```

---

## DV_MULTI_TRANSFER.py

### Basic Syntax

```bash
python DV_MULTI_TRANSFER.py ROOT_DIR SIM BASE_MODEL_NAME NEW_SIM NEW_L CLONE_NAME [OPTIONS]
```

### Purpose

Transfer learning allows you to:
- Adapt models trained on one simulation to another
- Fine-tune models for different interparticle separations
- Leverage pre-trained features for faster convergence

### Example Usage

```bash
# Transfer from TNG to Bolshoi
python DV_MULTI_TRANSFER.py /content/drive/MyDrive/ TNG base_tng_model BOL 10 transferred_model \
    --BATCH_SIZE 8 --EPOCHS 50 --LEARNING_RATE 1e-5
```

---

## Loss Functions Guide

### Recommended Loss Functions (Updated)

All the new improved loss functions are now available in the standard training scripts:

#### 1. `SCCE_Class_Penalty_Fixed` **(RECOMMENDED)**
```bash
python DV_MULTI_TRAIN.py /path/to/data TNG 10 4 16 SCCE_Class_Penalty_Fixed 512
```

#### 2. `SCCE_Proportion_Aware`
```bash
python DV_MULTI_TRAIN.py /path/to/data TNG 10 4 16 SCCE_Proportion_Aware 512
```

#### 3. `SCCE_Balanced_Class_Penalty`
```bash
python DV_MULTI_TRAIN.py /path/to/data TNG 10 4 16 SCCE_Balanced_Class_Penalty 512
```

### Standard Loss Functions

#### 4. `SCCE` (Safe Default)
```bash
python DV_MULTI_TRAIN.py /path/to/data TNG 10 4 16 SCCE 512
```

#### 5. `FOCAL_CCE` (Class Imbalance)
```bash
python DV_MULTI_TRAIN.py /path/to/data TNG 10 4 16 FOCAL_CCE 512 \
    --FOCAL_ALPHA 0.4 0.4 0.15 0.05 --FOCAL_GAMMA 2.0
```

#### 6. `DISCCE` (Dice + SCCE)
```bash
python DV_MULTI_TRAIN.py /path/to/data TNG 10 4 16 DISCCE 512
```

---

## Common Usage Examples

### New: Improved Loss Functions Shell Scripts

Three new shell scripts have been added to make it easy to use the improved loss functions:

#### 1. `train_with_improved_loss.sh` - Recommended Training
```bash
./train_with_improved_loss.sh
```
Uses `SCCE_Class_Penalty_Fixed` with optimized parameters for best results.

#### 2. `compare_loss_functions.sh` - Loss Function Testing  
```bash
./compare_loss_functions.sh
```
Easy configuration switching to test different improved loss functions.

#### 3. `fix_void_wall_bias.sh` - Quick Fix for Biased Models
```bash
./fix_void_wall_bias.sh MODEL_NAME
```
Retrain existing models with improved loss functions to fix void/wall bias.

### Quick Baseline Training

```bash
# Fast baseline with standard settings
python DV_MULTI_TRAIN.py /content/drive/MyDrive/ TNG 10 3 16 SCCE 512 \
    --BATCH_SIZE 16 --EPOCHS 100
```

### Production Quality Training

```bash
# High-quality model with all improvements
python DV_MULTI_TRAIN.py /content/drive/MyDrive/ TNG 10 4 32 SCCE_Class_Penalty_Fixed 512 \
    --BATCH_SIZE 8 --ATTENTION_UNET --BATCHNORM --DROPOUT 0.1 \
    --LAMBDA_CONDITIONING --LEARNING_RATE 5e-5 --EPOCHS 200
```

### Memory-Constrained Training

```bash
# Low-memory training
python DV_MULTI_TRAIN.py /content/drive/MyDrive/ TNG 10 3 8 SCCE 512 \
    --BATCH_SIZE 4 --LOW_MEM_FLAG --EPOCHS 100
```

### Experimental Features

```bash
# With extra inputs and RSD
python DV_MULTI_TRAIN.py /content/drive/MyDrive/ TNG 10 4 16 SCCE_Proportion_Aware 512 \
    --EXTRA_INPUTS g-r --ADD_RSD --ATTENTION_UNET --LAMBDA_CONDITIONING
```

### Binary Classification (Void vs Non-Void)

```bash
# Binary void detection
python DV_MULTI_TRAIN.py /content/drive/MyDrive/ TNG 10 4 16 BCE 512 \
    --BINARY_MASK --BATCH_SIZE 8 --ATTENTION_UNET
```

---

## Model Architecture Options

### Standard U-Net
```bash
# Basic U-Net architecture
python DV_MULTI_TRAIN.py /path/to/data TNG 10 4 16 SCCE 512
```

### Attention U-Net (Recommended)
```bash
# Enhanced attention mechanisms
python DV_MULTI_TRAIN.py /path/to/data TNG 10 4 16 SCCE_Class_Penalty_Fixed 512 \
    --ATTENTION_UNET
```

### Partial Convolution U-Net
```bash
# For handling missing data or survey masks
python DV_MULTI_TRAIN.py /path/to/data TNG 10 4 16 SCCE 512 \
    --USE_PCONV
```

### Lambda-Conditioned Models
```bash
# Multi-scale aware models
python DV_MULTI_TRAIN.py /path/to/data TNG 10 4 16 SCCE_Proportion_Aware 512 \
    --LAMBDA_CONDITIONING
```

---

## Troubleshooting

### Problem: Model overpredicts walls instead of voids

**Solution**: Use improved loss functions
```bash
python DV_MULTI_TRAIN.py /path/to/data TNG 10 4 16 SCCE_Class_Penalty_Fixed 512
```

### Problem: Poor minority class detection

**Solution**: Use focal loss or proportion-aware loss
```bash
python DV_MULTI_TRAIN.py /path/to/data TNG 10 4 16 FOCAL_CCE 512 \
    --FOCAL_ALPHA 0.5 0.3 0.15 0.05
```

### Problem: Training instability

**Solutions**:
1. Reduce learning rate: `--LEARNING_RATE 1e-5`
2. Add batch normalization: `--BATCHNORM`
3. Use dropout: `--DROPOUT 0.1`
4. Use standard SCCE loss: `SCCE`

### Problem: Out of memory

**Solutions**:
1. Reduce batch size: `--BATCH_SIZE 4`
2. Use low memory flag: `--LOW_MEM_FLAG`
3. Reduce model size: `DEPTH=3 FILTERS=8`
4. Reduce grid size: `GRID=256`

### Problem: Slow training

**Solutions**:
1. Increase batch size: `--BATCH_SIZE 32`
2. Remove unnecessary flags: Skip `--TENSORBOARD_FLAG`
3. Use multiprocessing: `--MULTI_FLAG`
4. Reduce epochs for testing: `--EPOCHS 50`

---

## Performance Monitoring

### TensorBoard Integration
```bash
# Enable TensorBoard logging
python DV_MULTI_TRAIN.py /path/to/data TNG 10 4 16 SCCE 512 \
    --TENSORBOARD_FLAG
```

### Custom Metrics

The scripts automatically compute:
- **Accuracy**: Overall pixel-wise accuracy
- **Matthews Correlation Coefficient (MCC)**: Balanced performance metric
- **F1 Scores**: Per-class and micro/macro averaged
- **Balanced Accuracy**: Accounts for class imbalance
- **Void Fraction**: Monitors void prediction percentage

### Debugging with diagnose_bias.py

```bash
# Analyze training logs for bias issues
python diagnose_bias.py /path/to/training_log.txt
```

---

## File Outputs

### Training Outputs
- **Model**: `MODEL_NAME.keras`
- **Weights**: `MODEL_NAME_weights.h5` 
- **History**: `MODEL_NAME_metrics.png`
- **Hyperparameters**: `MODEL_NAME_hyperparameters.txt`
- **Scores**: `MODEL_NAME_scores.csv`

### Prediction Outputs
- **Predictions**: `MODEL_NAME_Y_pred.npy`
- **Visualizations**: Various slice plots and metrics
- **Scores**: Comprehensive metrics in CSV format

---

## Best Practices

1. **Start with SCCE**: Begin with standard loss for baseline
2. **Use Attention**: `--ATTENTION_UNET` generally improves performance
3. **Monitor Training**: Watch loss curves and void fraction
4. **Validate on Different L**: Test model on multiple interparticle separations
5. **Save Checkpoints**: Models are automatically saved during training
6. **Compare Loss Functions**: Try multiple loss functions for your specific data
7. **Use Transfer Learning**: Leverage pre-trained models when possible

### Recommended Training Pipeline

1. **Quick Test**: Train with SCCE, small model, few epochs
2. **Baseline**: Train with SCCE_Class_Penalty_Fixed, standard settings
3. **Optimization**: Experiment with different loss functions and architectures
4. **Production**: Final training with best settings, full epochs

The standard training scripts provide flexible, single-scale training with all the latest loss function improvements integrated!
