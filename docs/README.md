# DeepVoid Documentation

Complete documentation for the DeepVoid cosmic void detection project.

## Documentation Index

### Getting Started
- **[Standard Scripts Usage Guide](STANDARD_SCRIPTS_USAGE_GUIDE.md)** - Complete guide for DV_MULTI_TRAIN.py, DV_MULTI_TRANSFER.py, and attention_test.py
- **[Curricular Training Guide](CURRICULAR_USAGE_GUIDE.md)** - Multi-scale progressive training with curricular.py

### Advanced Features  
- **[Loss Function Improvements](LOSS_FUNCTION_IMPROVEMENTS.md)** - New loss functions that fix void/wall prediction bias
- **Model Interpretability Analysis** - Understand what your neural networks learn with feature maps and attention visualizations

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

**Model Interpretability Analysis**:
```bash
# Analyze a single trained model
python model_interpretability.py path/to/model.h5 path/to/test_data.h5

# Comprehensive analysis with custom parameters
python model_interpretability.py model.h5 data.h5 \
    --output_dir ./analysis_results \
    --num_samples 20 \
    --slice_indices 32 48 64 \
    --correlation_threshold 0.3 \
    --dpi 150

# Quick feature map extraction only
python model_interpretability.py model.h5 data.h5 --feature_maps_only
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

### Model Interpretability Analysis

The `model_interpretability.py` script provides comprehensive analysis tools to understand what your DeepVoid neural networks learn during cosmic void detection. This is crucial for validating model behavior and gaining scientific insights.

#### Features

**🔍 Feature Activation Maps**
- Extract and visualize feature maps from all convolutional layers
- 3D visualization with interactive slicing and matplotlib integration
- Automatic layer detection and naming
- Statistical analysis of activation patterns

**👁️ Attention Map Analysis** (for Attention U-Net models)
- Extract attention gates from encoder-decoder connections
- Visualize where the model focuses during void detection
- Compare attention patterns across different scales
- Quantify attention distribution and focus areas

**📊 Void-Feature Correlation Analysis**
- Correlate model predictions with feature activations
- Identify which features are most predictive of voids
- Generate correlation heatmaps and statistical summaries
- Find optimal correlation thresholds for feature selection

**🎨 Publication-Ready Visualizations**
- High-DPI figures suitable for papers and presentations
- Customizable colormaps, layouts, and annotations
- Automatic scaling and normalization
- Export in multiple formats (PNG, PDF, SVG)

#### Usage Examples

**Basic Analysis**:
```bash
# Analyze a trained curricular model with specific lambda
python model_interpretability.py /content/drive/MyDrive/ TNG_curricular_model_name 10
```

**Advanced Analysis with Custom Parameters**:
```bash
python model_interpretability.py /content/drive/MyDrive/ TNG_curricular_SCCE_Class_Penalty_Fixed_D4_F16_RSD_attention 10 \
    --MAX_SAMPLES 50 \
    --SLICE_IDX 64 \
    --DPI 300 \
    --SAVE_ALL \
    --EXTRA_INPUTS g-r \
    --ADD_RSD \
    --USE_OVERLAPPING_SUBCUBES \
    --RSD_PRESERVING_ROTATIONS
```

**Quick Feature Extraction for Multiple Models**:
```bash
# Batch analyze multiple curricular models
for model_name in TNG_curricular_model1 TNG_curricular_model2; do
    python model_interpretability.py /content/drive/MyDrive/ "$model_name" 10 \
        --MAX_SAMPLES 20 --SAVE_ALL
done
```

#### Output Structure

The script generates organized output directories:

```
interpretability_results/
├── feature_maps/
│   ├── layer_conv3d_1_sample_0.png
│   ├── layer_conv3d_2_sample_0.png
│   └── ...
├── attention_maps/          # (Attention U-Net only)
│   ├── attention_gate_1.png
│   └── attention_gate_2.png
├── correlations/
│   ├── void_correlation_heatmap.png
│   ├── correlation_summary.txt
│   └── high_correlation_features.csv
└── analysis_log.txt
```

#### Key Parameters

- `--MAX_SAMPLES`: Number of test samples to analyze (default: 3)
- `--MAX_FILTERS`: Maximum filters to visualize per layer (default: 16)
- `--SLICE_IDX`: Which 3D slice to visualize (default: middle slice)
- `--DPI`: Figure resolution for publications (default: 300)
- `--SAVE_ALL`: Save all analysis plots (feature maps, attention, etc.)
- `--EXTRA_INPUTS`: Extra inputs used in training (g-r, r_flux_density)
- `--ADD_RSD`: Model trained with redshift-space distortions
- `--LAMBDA_CONDITIONING`: Model uses lambda conditioning
- `--USE_OVERLAPPING_SUBCUBES`: Use overlapping subcubes (more diverse samples, uses more memory)
- `--RSD_PRESERVING_ROTATIONS`: Use RSD-preserving rotations (auto-enabled for RSD models)
- `--EXTRA_AUGMENTATION`: Use extra data augmentation (match training settings)

#### Scientific Applications

**Model Validation**:
- Verify that models focus on relevant cosmic structures
- Ensure attention patterns align with physical expectations
- Identify potential overfitting or dataset biases

**Feature Discovery**:
- Understand which density patterns predict voids
- Compare feature importance across different scales
- Guide physics-informed model improvements

**Transfer Learning Insights**:
- Analyze how features transfer between different simulations
- Understand domain adaptation in cosmological models
- Optimize transfer learning strategies

#### Technical Requirements

- TensorFlow/Keras models (`.h5` or `.keras` format)
- Test data in `.fvol` format (same as training data)
- Sufficient memory for feature map extraction (adjust `num_samples` if needed)
- Matplotlib and scipy for visualizations and statistics

#### Troubleshooting

**Memory Issues**:
```bash
# Reduce number of samples and focus on specific layers
python model_interpretability.py model.h5 data.h5 --num_samples 5
```

**Large Models**:
```bash
# Extract feature maps only to reduce computation
python model_interpretability.py model.h5 data.h5 --feature_maps_only
```

**Custom Data Format**:
The script uses `.fvol` files exactly like the training scripts (`curricular.py`, `DV_MULTI_TRAIN.py`). Data files should be in the same format and location as used during training.

---

**Last Updated**: August 2025  
**Version**: 2.0 with improved loss functions and bias fixes
