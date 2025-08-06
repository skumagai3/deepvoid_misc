# DeepVoid Curricular Training - Usage Guide

## Overview

This guide covers the usage of `curricular.py` for training and `curricular_pred.py` for prediction with the DeepVoid cosmic void detection models.

## Table of Contents

1. [Curricular Training (`curricular.py`)](#curricular-training)
2. [Prediction (`curricular_pred.py`)](#prediction)
3. [Loss Functions Guide](#loss-functions-guide)
4. [Common Usage Examples](#common-usage-examples)
5. [Troubleshooting](#troubleshooting)

---

## Curricular Training

### Basic Syntax

```bash
python curricular.py ROOT_DIR DEPTH FILTERS LOSS [OPTIONS]
```

### Required Arguments

- `ROOT_DIR`: Root directory for the project (e.g., `/content/drive/MyDrive/`)
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
| `--USE_ATTENTION` | False | Use attention U-Net architecture |
| `--LAMBDA_CONDITIONING` | False | Use lambda conditioning in the model |
| `--N_EPOCHS_PER_INTER_SEP` | 50 | Epochs to train for each interparticle separation |
| `--EXTRA_INPUTS` | None | Additional inputs ('g-r' or 'r_flux_density') |
| `--ADD_RSD` | False | Add Redshift Space Distortion to inputs |

### Training Process

Curricular training progressively trains on different interparticle separations:
1. Starts with lowest separation (0.33 Mpc/h)
2. Progressively increases: 0.33 → 3 → 5 → 7 → 10 Mpc/h
3. Trains for `N_EPOCHS_PER_INTER_SEP` epochs at each level
4. Saves checkpoints at each stage

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

### 🧪 Experimental Loss Functions

#### 7. `FOCAL_CCE`
- **Features**: Focal loss for handling class imbalance
- **Parameters**: `alpha=[0.4, 0.4, 0.15, 0.05]`, `gamma=2.0`

#### 8. `DISCCE`
- **Features**: Combines SCCE with Dice loss
- **Parameters**: `cce_weight=1.0`, `dice_weight=1.0`

---

## Common Usage Examples

### Basic Training (Recommended)

```bash
# Best balanced approach
python curricular.py /content/drive/MyDrive/ 4 16 SCCE_Class_Penalty_Fixed \
    --BATCH_SIZE 8 --USE_ATTENTION --LAMBDA_CONDITIONING --L_VAL 10

# Proportion-aware training
python curricular.py /content/drive/MyDrive/ 4 16 SCCE_Proportion_Aware \
    --BATCH_SIZE 8 --USE_ATTENTION --LAMBDA_CONDITIONING --L_VAL 10

# Safe fallback (standard loss)
python curricular.py /content/drive/MyDrive/ 4 16 SCCE \
    --BATCH_SIZE 8 --USE_ATTENTION --LAMBDA_CONDITIONING --L_VAL 10
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
