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
# Analyze a trained model with test data
python model_interpretability.py /path/to/trained_model.h5 /path/to/test_data.h5
```

**Advanced Analysis with Custom Parameters**:
```bash
python model_interpretability.py model.h5 data.h5 \
    --output_dir ./interpretability_results \
    --num_samples 50 \
    --slice_indices 16 32 48 64 80 \
    --correlation_threshold 0.25 \
    --dpi 300 \
    --feature_maps_only
```

**Quick Feature Extraction for Multiple Models**:
```bash
# Batch analyze multiple models
for model in models/*.h5; do
    python model_interpretability.py "$model" test_data.h5 \
        --output_dir "analysis_$(basename $model .h5)" \
        --num_samples 20
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

- `--num_samples`: Number of test samples to analyze (default: 10)
- `--slice_indices`: Which 3D slices to visualize (default: center slices)
- `--correlation_threshold`: Minimum correlation to report (default: 0.3)
- `--output_dir`: Where to save analysis results
- `--feature_maps_only`: Skip correlation analysis for faster execution
- `--dpi`: Figure resolution for publications (default: 150)

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

- TensorFlow/Keras models (`.h5` format)
- Test data in HDF5 format with `density` and `labels` datasets
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
The script expects HDF5 files with `density` and `labels` datasets. If your data format differs, modify the data loading section in the script.

---

**Last Updated**: August 2025  
**Version**: 2.0 with improved loss functions and bias fixes
