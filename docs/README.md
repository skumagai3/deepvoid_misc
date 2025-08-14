# DeepVoid Documentation

Complete documentation for the DeepVoid cosmic void detection project.

## Documentation Index

### Getting Started
- **[Standard Scripts Usage Guide](STANDARD_SCRIPTS_USAGE_GUIDE.md)** - Complete guide for DV_MULTI_TRAIN.py, DV_MULTI_TRANSFER.py, and attention_test.py
- **[Curricular Training Guide](CURRICULAR_USAGE_GUIDE.md)** - Multi-scale progressive training with curricular.py

### Advanced Features  
- **[Loss Function Improvements](LOSS_FUNCTION_IMPROVEMENTS.md)** - New loss functions that fix void/wall prediction bias

### Quick Reference

#### New Improved Loss Functions (2025 Update)
1. **`SCCE_Class_Penalty_Fixed`** **(RECOMMENDED)** - Best balanced approach
2. **`SCCE_Proportion_Aware`** - Maintains target class proportions  
3. **`SCCE_Balanced_Class_Penalty`** - Alternative balanced approach
4. **`SCCE`** - Standard safe fallback

#### Key Features Available
- **Attention U-Net** - Improved feature extraction with attention gates
- **Lambda Conditioning** - Scale-aware training and prediction
- **Redshift Space Distortions** - Realistic observational effects
- **Curricular Training** - Progressive multi-scale training with 4 validation strategies
- **Extra Inputs** - Galaxy colors and flux density integration
- **Transfer Learning** - Efficient adaptation between scales
- **Bias Diagnosis** - Automated tools to detect prediction issues
- **Advanced Preprocessing** - 4 preprocessing methods for difficult data
- **Learning Rate Warmup** - Gradual learning rate increase for stability
- **Memory Optimization** - Configurable overlapping subcubes and data augmentation

#### Quick Start Commands

**Standard Training (Recommended)**:
```bash
python DV_MULTI_TRAIN.py /content/drive/MyDrive/ TNG 0.33 4 16 SCCE_Class_Penalty_Fixed 512 \
    --ATTENTION_UNET --LAMBDA_CONDITIONING --BATCH_SIZE 8
```

**Curricular Training**:
```bash
python curricular.py /content/drive/MyDrive/ 4 16 SCCE_Class_Penalty_Fixed \
    --USE_ATTENTION --LAMBDA_CONDITIONING --BATCH_SIZE 8 \
    --VALIDATION_STRATEGY gradual --PREPROCESSING robust --WARMUP_EPOCHS 10
```

**Transfer Learning**:
```bash
python DV_MULTI_TRANSFER.py ROOT_DIR MODEL_NAME DENSITY_FILE TL_TYPE SIM GRID \
    --LOSS SCCE_Class_Penalty_Fixed
```

### New Shell Scripts
- `train_with_improved_loss.sh` - Ready-to-use training with recommended settings
- `compare_loss_functions.sh` - Easy testing of different loss functions  
- `fix_void_wall_bias.sh` - Quick fix for existing biased models

### Troubleshooting
If your model predicts wall instead of void heavily, use:
```bash
python diagnose_bias.py path/to/training.log
```

### Full Guides
Each guide contains comprehensive usage examples, parameter explanations, and troubleshooting tips:

1. **[STANDARD_SCRIPTS_USAGE_GUIDE.md](STANDARD_SCRIPTS_USAGE_GUIDE.md)** - 379 lines of detailed documentation
2. **[CURRICULAR_USAGE_GUIDE.md](CURRICULAR_USAGE_GUIDE.md)** - 270 lines covering progressive training
3. **[LOSS_FUNCTION_IMPROVEMENTS.md](LOSS_FUNCTION_IMPROVEMENTS.md)** - Technical details on bias fixes

---

**Last Updated**: August 2025  
**Version**: 2.0 with improved loss functions and bias fixes
