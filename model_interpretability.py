#!/usr/bin/env python3
"""
Model Interpretability Analysis for DeepVoid Networks

This script provides comprehensive interpretability analysis for trained DeepVoid models,
including feature activation maps, attention visualizations, and void-specific analysis.

Usage:
python model_interpretability.py ROOT_DIR MODEL_NAME L_ANALYSIS --PREPROCESSING method [options]

Example:
python model_interpretability.py /content/drive/MyDrive/ TNG_curricular_SCCE_Proportion_Aware_D4_F16_RSD_attention_g-r_2025-08-18_17-16-07 10 --PREPROCESSING standard --SAVE_ALL --EXTRA_INPUTS g-r --ADD_RSD

IMPORTANT: You MUST specify --PREPROCESSING to match the method used during training!
Common preprocessing methods:
- standard: Standard normalization (mean=0, std=1) 
- robust: Robust scaling using median and IQR
- log_transform: Log transformation + scaling
- clip_extreme: Clip extreme values + scaling

This will generate comprehensive interpretability analysis including:
- Feature activation maps across all network layers
- Attention maps (for attention models)
- Void-specific activation analysis
- Feature evolution through network depth
- Correlation analysis between activations and cosmic structures

Data Format: Uses .fvol files exactly like curricular.py and curricular_pred.py
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
import gc
from datetime import datetime
from scipy.ndimage import zoom
from sklearn.metrics import matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
sys.path.append('.')
import NETS_LITE as nets
try:
    import volumes  # For .fvol file loading
except ImportError:
    print("Warning: volumes module not found. Data loading may fail.")
    volumes = None

print('>>> Running model_interpretability.py')
print('TensorFlow version:', tf.__version__)

# Set up GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
print('GPUs available:', gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#================================================================
# Parse command line arguments
#================================================================
parser = argparse.ArgumentParser(description='Model interpretability analysis for DeepVoid networks.')
required = parser.add_argument_group('required arguments')
required.add_argument('ROOT_DIR', type=str, help='Root directory for the project.')
required.add_argument('MODEL_NAME', type=str, help='Name of the trained model (without file extension).')
required.add_argument('L_ANALYSIS', type=str, help='Interparticle separation for analysis (e.g., "10").')

optional = parser.add_argument_group('optional arguments')
optional.add_argument('--EXTRA_INPUTS', type=str, default=None,
                      choices=['g-r', 'r_flux_density'],
                      help='Name of additional inputs used in training.')
optional.add_argument('--ADD_RSD', action='store_true',
                      help='Use RSD data files (same as training).')
optional.add_argument('--LAMBDA_CONDITIONING', action='store_true',
                      help='Model uses lambda conditioning.')
optional.add_argument('--PREPROCESSING', type=str, default=None,
                      choices=['standard', 'robust', 'log_transform', 'clip_extreme'],
                      help='Preprocessing method used in training. REQUIRED to match training preprocessing exactly.')
optional.add_argument('--MAX_SAMPLES', type=int, default=3,
                      help='Maximum number of samples to analyze. Default is 3.')
optional.add_argument('--MAX_FILTERS', type=int, default=16,
                      help='Maximum number of filters to show per layer. Default is 16.')
optional.add_argument('--USE_OVERLAPPING_SUBCUBES', action='store_true',
                      help='Use overlapping subcubes for more diverse samples (uses more memory).')
optional.add_argument('--RSD_PRESERVING_ROTATIONS', action='store_true',
                      help='Use RSD-preserving rotations (for models trained with RSD data).')
optional.add_argument('--EXTRA_AUGMENTATION', action='store_true',
                      help='Use extra data augmentation (for models trained with extra augmentation).')
optional.add_argument('--SAVE_ALL', action='store_true',
                      help='Save all analysis plots (feature maps, attention, evolution, etc.).')
optional.add_argument('--SLICE_IDX', type=int, default=None,
                      help='Specific Z-slice index to analyze (None = middle slice).')
optional.add_argument('--DPI', type=int, default=300,
                      help='DPI for saved figures. Default is 300.')

args = parser.parse_args()

# Validate required arguments
if args.PREPROCESSING is None:
    print("ERROR: --PREPROCESSING is required!")
    print("You must specify the preprocessing method that was used during training.")
    print("Common options:")
    print("  --PREPROCESSING standard    # Standard normalization (mean=0, std=1)")
    print("  --PREPROCESSING robust      # Robust scaling using median and IQR")
    print("  --PREPROCESSING log_transform  # Log transformation + scaling")
    print("  --PREPROCESSING clip_extreme   # Clip extreme values + scaling")
    print("\nThis is critical for meaningful results - preprocessing must match training exactly!")
    sys.exit(1)

ROOT_DIR = args.ROOT_DIR
MODEL_NAME = args.MODEL_NAME
L_ANALYSIS = args.L_ANALYSIS
EXTRA_INPUTS = args.EXTRA_INPUTS
ADD_RSD = args.ADD_RSD
LAMBDA_CONDITIONING = args.LAMBDA_CONDITIONING
PREPROCESSING = args.PREPROCESSING  # Will be validated above
MAX_SAMPLES = args.MAX_SAMPLES
MAX_FILTERS = args.MAX_FILTERS
USE_OVERLAPPING_SUBCUBES = args.USE_OVERLAPPING_SUBCUBES
RSD_PRESERVING_ROTATIONS = args.RSD_PRESERVING_ROTATIONS
EXTRA_AUGMENTATION = args.EXTRA_AUGMENTATION
SAVE_ALL = args.SAVE_ALL
SLICE_IDX = args.SLICE_IDX
DPI = args.DPI

print(f'Analysis parameters: MODEL={MODEL_NAME}, L={L_ANALYSIS}, SAMPLES={MAX_SAMPLES}')
print(f'Extra inputs: {EXTRA_INPUTS}, RSD: {ADD_RSD}, Preprocessing: {PREPROCESSING}')
print(f'Data loading: Overlapping={USE_OVERLAPPING_SUBCUBES}, RSD-preserving={RSD_PRESERVING_ROTATIONS}, Extra-aug={EXTRA_AUGMENTATION}')

#================================================================
# Auto-detect model parameters from model name
#================================================================
print('Auto-detecting model parameters from model name...')
model_parts = MODEL_NAME.split('_')

# Auto-detect RSD usage
if not args.ADD_RSD and 'RSD' in model_parts:
    ADD_RSD = True
    print('Auto-detected: Model trained with RSD data')

# Auto-detect extra inputs
if args.EXTRA_INPUTS is None:
    if 'g-r' in model_parts:
        EXTRA_INPUTS = 'g-r'
        print('Auto-detected: Model uses g-r color as extra input')
    elif 'r-flux-density' in model_parts or 'r_flux_density' in model_parts:
        EXTRA_INPUTS = 'r_flux_density'
        print('Auto-detected: Model uses r-band flux density as extra input')

# Auto-detect lambda conditioning
if not args.LAMBDA_CONDITIONING and 'lambda' in model_parts:
    LAMBDA_CONDITIONING = True
    print('Auto-detected: Model uses lambda conditioning')

# Auto-detect attention architecture
USE_ATTENTION = 'attention' in model_parts
print(f'Auto-detected: Attention architecture = {USE_ATTENTION}')

# Auto-detect and enable RSD-preserving rotations if model was trained with RSD
if ADD_RSD and not args.RSD_PRESERVING_ROTATIONS:
    RSD_PRESERVING_ROTATIONS = True
    print('Auto-enabled: RSD-preserving rotations (model trained with RSD data)')

# Auto-detect preprocessing method from model name if not specified
if args.PREPROCESSING is None:
    if 'robust' in MODEL_NAME.lower():
        PREPROCESSING = 'robust'
        print('Auto-detected: Robust preprocessing')
    elif 'log' in MODEL_NAME.lower():
        PREPROCESSING = 'log_transform'
        print('Auto-detected: Log transform preprocessing')
    elif 'clip' in MODEL_NAME.lower():
        PREPROCESSING = 'clip_extreme'
        print('Auto-detected: Clip extreme preprocessing')
    elif 'standard' in MODEL_NAME.lower() or 'std' in MODEL_NAME.lower():
        PREPROCESSING = 'standard'
        print('Auto-detected: Standard preprocessing')
    else:
        # Still require explicit specification if not auto-detected
        print("ERROR: Could not auto-detect preprocessing method from model name!")
        print("You must specify --PREPROCESSING explicitly.")
        print("Common options:")
        print("  --PREPROCESSING standard    # Standard normalization (mean=0, std=1)")
        print("  --PREPROCESSING robust      # Robust scaling using median and IQR")
        print("  --PREPROCESSING log_transform  # Log transformation + scaling")
        print("  --PREPROCESSING clip_extreme   # Clip extreme values + scaling")
        print("\nThis is critical for meaningful results - preprocessing must match training exactly!")
        sys.exit(1)
else:
    PREPROCESSING = args.PREPROCESSING

print(f'Final parameters: RSD={ADD_RSD}, EXTRA_INPUTS={EXTRA_INPUTS}, LAMBDA_CONDITIONING={LAMBDA_CONDITIONING}, ATTENTION={USE_ATTENTION}')
print(f'Data loading: Overlapping={USE_OVERLAPPING_SUBCUBES}, RSD-preserving={RSD_PRESERVING_ROTATIONS}, Extra-aug={EXTRA_AUGMENTATION}')
print(f'Preprocessing: {PREPROCESSING} (CRITICAL: must match training preprocessing exactly!)')

#================================================================
# Set up paths and parameters
#================================================================
DATA_PATH = ROOT_DIR + 'data/TNG/'
MODEL_PATH = ROOT_DIR + 'models/'
ANALYSIS_PATH = ROOT_DIR + f'interpretability/{MODEL_NAME}/'

# Ensure output directory exists
os.makedirs(ANALYSIS_PATH, exist_ok=True)
print(f'Analysis results will be saved to: {ANALYSIS_PATH}')

# Model parameters
BoxSize = 205.0  # Mpc/h
class_labels = ['void', 'wall', 'fila', 'halo']
N_CLASSES = 4
GRID = 512
SUBGRID = 128
OFF = 64
th = 0.65
sig = 2.4

FILE_MASK = DATA_PATH + f'TNG300-3-Dark-mask-Nm={GRID}-th={th}-sig={sig}.fvol'

# Data file mapping
data_info = {
    '0.33': 'DM_DEN_snap99_Nm=512.fvol',
    '3': 'subs1_mass_Nm512_L3_d_None_smooth.fvol',
    '5': 'subs1_mass_Nm512_L5_d_None_smooth.fvol',
    '7': 'subs1_mass_Nm512_L7_d_None_smooth.fvol',
    '10': 'subs1_mass_Nm512_L10_d_None_smooth.fvol',
}

if ADD_RSD:
    data_info['0.33'] = 'DM_DEN_snap99_perturbed_Nm=512.fvol'
    data_info['3'] = 'subs1_mass_Nm512_L3_RSD.fvol'
    data_info['5'] = 'subs1_mass_Nm512_L5_RSD.fvol'
    data_info['7'] = 'subs1_mass_Nm512_L7_RSD.fvol'
    data_info['10'] = 'subs1_mass_Nm512_L10_RSD.fvol'

# Extra inputs mapping
EXTRA_INPUTS_INFO = {}
if EXTRA_INPUTS == 'g-r':
    EXTRA_INPUTS_INFO = {
        '0.33': 'subs1_g-r_Nm512_L3.fvol',
        '3': 'subs1_g-r_Nm512_L3.fvol',
        '5': 'subs1_g-r_Nm512_L5.fvol',
        '7': 'subs1_g-r_Nm512_L7.fvol',
        '10': 'subs1_g-r_Nm512_L10.fvol',
    }
elif EXTRA_INPUTS == 'r_flux_density':
    EXTRA_INPUTS_INFO = {
        '0.33': 'subs1_r_flux_density_Nm512_L3.fvol',
        '3': 'subs1_r_flux_density_Nm512_L3.fvol',
        '5': 'subs1_r_flux_density_Nm512_L5.fvol',
        '7': 'subs1_r_flux_density_Nm512_L7.fvol',
        '10': 'subs1_r_flux_density_Nm512_L10.fvol',
    }

if ADD_RSD and EXTRA_INPUTS is not None:
    for key in EXTRA_INPUTS_INFO:
        EXTRA_INPUTS_INFO[key] = EXTRA_INPUTS_INFO[key].replace('.fvol', '_RSD.fvol')

#================================================================
# Custom objects for model loading
#================================================================
CUSTOM_OBJECTS = {
    'MCC_keras': nets.MCC_keras,
    'F1_micro_keras': nets.F1_micro_keras,
    'void_F1_keras': nets.void_F1_keras,
    'SCCE_Dice_loss': nets.SCCE_Dice_loss,
    'categorical_focal_loss': nets.categorical_focal_loss,
    'SCCE_void_penalty': nets.SCCE_void_penalty,
    'SCCE_Class_Penalty': nets.SCCE_Class_Penalty,
    'SCCE_Balanced_Class_Penalty': nets.SCCE_Balanced_Class_Penalty,
    'SCCE_Class_Penalty_Fixed': nets.SCCE_Class_Penalty_Fixed,
    'SCCE_Proportion_Aware': nets.SCCE_Proportion_Aware,
    'VoidFractionMonitor': nets.VoidFractionMonitor,
    'RobustModelCheckpoint': nets.RobustModelCheckpoint,
    'Cast': tf.cast
}

#================================================================
# Feature extraction functions
#================================================================
def extract_feature_maps(model, input_data, layer_names=None, max_samples=MAX_SAMPLES):
    """
    Extract feature activation maps from intermediate layers.
    """
    print(f'Extracting feature maps from {len(input_data)} samples...')
    
    if layer_names is None:
        # Auto-select key layers
        layer_names = []
        for layer in model.layers:
            layer_name = layer.name.lower()
            if any(keyword in layer_name for keyword in ['encoder_block', 'bottleneck', 'decoder_block']):
                if 'activation' not in layer_name and 'batch_norm' not in layer_name:
                    layer_names.append(layer.name)
        
        # Also include some conv layers
        for layer in model.layers:
            if 'conv3d' in layer.name.lower() and len(layer_names) < 20:
                layer_names.append(layer.name)
    
    print(f'Selected {len(layer_names)} layers for feature extraction:')
    for name in layer_names[:10]:  # Print first 10
        print(f'  - {name}')
    if len(layer_names) > 10:
        print(f'  ... and {len(layer_names) - 10} more')
    
    # Create feature extraction model
    try:
        layer_outputs = []
        for name in layer_names:
            try:
                layer_outputs.append(model.get_layer(name).output)
            except ValueError:
                print(f'Warning: Layer {name} not found, skipping')
                continue
        
        if not layer_outputs:
            raise ValueError("No valid layers found for feature extraction")
        
        feature_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
        print(f'Created feature extraction model with {len(layer_outputs)} outputs')
        
    except Exception as e:
        print(f'Error creating feature model: {e}')
        return {}
    
    # Extract features
    input_batch = input_data[:max_samples] if len(input_data) > max_samples else input_data
    print(f'Processing {len(input_batch)} samples...')
    
    try:
        features = feature_model.predict(input_batch, verbose=0)
        print('Feature extraction completed successfully')
    except Exception as e:
        print(f'Error during feature extraction: {e}')
        return {}
    
    # Organize results
    feature_dict = {}
    valid_layer_names = [name for name in layer_names if name in [layer.name for layer in model.layers]]
    
    if isinstance(features, list):
        for i, layer_name in enumerate(valid_layer_names):
            if i < len(features):
                feature_dict[layer_name] = features[i]
    else:
        if len(valid_layer_names) > 0:
            feature_dict[valid_layer_names[0]] = features
    
    print(f'Extracted features from {len(feature_dict)} layers')
    return feature_dict

def extract_attention_maps(model, input_data, max_samples=MAX_SAMPLES):
    """
    Extract attention gate outputs from Attention U-Net.
    """
    print('Extracting attention maps...')
    
    # Find attention-related layers
    attention_layers = []
    for layer in model.layers:
        layer_name = layer.name.lower()
        if 'multiply' in layer_name or 'attention' in layer_name:
            attention_layers.append(layer.name)
    
    if not attention_layers:
        print("No attention layers found. This may not be an Attention U-Net.")
        return {}
    
    print(f'Found {len(attention_layers)} attention-related layers:')
    for name in attention_layers:
        print(f'  - {name}')
    
    try:
        attention_outputs = [model.get_layer(name).output for name in attention_layers]
        attention_model = tf.keras.Model(inputs=model.input, outputs=attention_outputs)
        
        input_batch = input_data[:max_samples] if len(input_data) > max_samples else input_data
        attention_maps = attention_model.predict(input_batch, verbose=0)
        
        # Organize by depth level
        attention_dict = {}
        for i, layer_name in enumerate(attention_layers):
            attention_dict[layer_name] = attention_maps[i] if isinstance(attention_maps, list) else attention_maps
        
        print(f'Extracted attention maps from {len(attention_dict)} layers')
        return attention_dict
        
    except Exception as e:
        print(f'Error extracting attention maps: {e}')
        return {}

#================================================================
# Visualization functions
#================================================================
def plot_feature_maps_3d(feature_dict, sample_idx=0, slice_idx=None, max_filters=MAX_FILTERS, 
                         save_path=None, void_regions=None):
    """
    Plot 3D feature activation maps as 2D slices.
    """
    print(f'Creating feature maps plot for sample {sample_idx}...')
    
    if not feature_dict:
        print("No feature maps to plot")
        return None
    
    # Select representative layers
    layer_names = list(feature_dict.keys())
    # Try to get a good spread of layers
    if len(layer_names) > 8:
        # Sample layers across the network
        indices = np.linspace(0, len(layer_names)-1, 8, dtype=int)
        selected_layers = [layer_names[i] for i in indices]
    else:
        selected_layers = layer_names
    
    n_layers = len(selected_layers)
    
    fig, axes = plt.subplots(n_layers, max_filters, figsize=(max_filters*2, n_layers*2))
    if n_layers == 1:
        axes = axes[np.newaxis, :]
    
    for layer_idx, layer_name in enumerate(selected_layers):
        if layer_name not in feature_dict:
            continue
            
        features = feature_dict[layer_name]
        if len(features) <= sample_idx:
            continue
            
        feature_map = features[sample_idx]  # Shape: (H, W, D, C)
        
        if slice_idx is None:
            slice_idx_use = feature_map.shape[2] // 2  # Middle slice
        else:
            slice_idx_use = min(slice_idx, feature_map.shape[2] - 1)
        
        # Select slice
        feature_slice = feature_map[:, :, slice_idx_use, :]  # Shape: (H, W, C)
        n_filters = min(feature_slice.shape[-1], max_filters)
        
        for filter_idx in range(n_filters):
            if filter_idx >= feature_slice.shape[-1]:
                break
                
            ax = axes[layer_idx, filter_idx] if n_layers > 1 else axes[filter_idx]
            
            # Plot activation map
            activation = feature_slice[:, :, filter_idx]
            im = ax.imshow(activation, cmap='viridis', aspect='equal')
            
            # Overlay void regions if provided
            if void_regions is not None and void_regions.shape[:2] == activation.shape:
                ax.contour(void_regions, levels=[0.5], colors='red', linewidths=1, alpha=0.7)
            
            if filter_idx == 0:
                ax.set_ylabel(f'{layer_name}\n(Filter {filter_idx})', fontsize=8)
            else:
                ax.set_title(f'Filter {filter_idx}', fontsize=8)
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar for first filter of each layer
            if filter_idx == 0:
                plt.colorbar(im, ax=ax, shrink=0.6)
        
        # Hide unused subplots
        for filter_idx in range(n_filters, max_filters):
            ax = axes[layer_idx, filter_idx] if n_layers > 1 else axes[filter_idx]
            ax.set_visible(False)
    
    plt.suptitle(f'Feature Activation Maps - Sample {sample_idx}, Slice {slice_idx_use}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f'Feature maps saved to {save_path}')
    
    return fig

def plot_attention_maps_3d(attention_dict, original_input, void_mask=None, 
                          sample_idx=0, slice_idx=None, save_path=None):
    """
    Plot 3D attention maps overlaid on original input.
    """
    print(f'Creating attention maps plot for sample {sample_idx}...')
    
    if not attention_dict:
        print("No attention maps to plot")
        return None
    
    n_attention_levels = len(attention_dict)
    
    fig, axes = plt.subplots(2, n_attention_levels, figsize=(n_attention_levels*4, 8))
    if n_attention_levels == 1:
        axes = axes[:, np.newaxis]
    
    # Get original input slice
    if slice_idx is None:
        slice_idx_use = original_input.shape[2] // 2
    else:
        slice_idx_use = slice_idx
    
    original_slice = original_input[sample_idx, :, :, slice_idx_use, 0]
    
    for i, (level_name, attention_map) in enumerate(attention_dict.items()):
        # Top row: Original input with attention overlay
        ax1 = axes[0, i] if n_attention_levels > 1 else axes[0]
        
        # Show original input as background
        ax1.imshow(original_slice, cmap='gray', alpha=0.7)
        
        # Get attention slice
        attention_data = attention_map[sample_idx]
        if len(attention_data.shape) == 4:  # (H, W, D, C)
            attention_slice_idx = min(slice_idx_use // max(1, 2**i), attention_data.shape[2] - 1)
            attention_slice = attention_data[:, :, attention_slice_idx, :]
            # Average across channels if multiple
            if attention_slice.shape[-1] > 1:
                attention_slice = np.mean(attention_slice, axis=-1)
            else:
                attention_slice = attention_slice[:, :, 0]
        else:
            attention_slice = attention_data
        
        # Resize attention to match original if needed
        if attention_slice.shape != original_slice.shape:
            scale_factors = [original_slice.shape[j] / attention_slice.shape[j] for j in range(2)]
            attention_slice = zoom(attention_slice, scale_factors, order=1)
        
        im1 = ax1.imshow(attention_slice, cmap='hot', alpha=0.6)
        ax1.set_title(f'{level_name}\nInput + Attention')
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Bottom row: Pure attention map
        ax2 = axes[1, i] if n_attention_levels > 1 else axes[1]
        im2 = ax2.imshow(attention_slice, cmap='hot')
        ax2.set_title(f'{level_name}\nPure Attention')
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # Overlay void regions if provided
        if void_mask is not None:
            try:
                if len(void_mask.shape) == 4:
                    void_slice = void_mask[sample_idx, :, :, slice_idx_use]
                elif len(void_mask.shape) == 3:
                    void_slice = void_mask[:, :, slice_idx_use]
                else:
                    void_slice = void_mask
                
                ax1.contour(void_slice, levels=[0.5], colors='cyan', linewidths=2, alpha=0.8)
                ax2.contour(void_slice, levels=[0.5], colors='cyan', linewidths=2, alpha=0.8)
            except Exception as e:
                print(f'Warning: Could not overlay void mask: {e}')
    
    plt.suptitle(f'Attention Maps - Sample {sample_idx}, Slice {slice_idx_use}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f'Attention maps saved to {save_path}')
    
    return fig

def analyze_void_activation_correlation(feature_dict, void_mask, sample_idx=0):
    """
    Analyze which filters are most correlated with void regions.
    """
    print(f'Analyzing void-activation correlations for sample {sample_idx}...')
    
    correlations = {}
    
    for layer_name, features in feature_dict.items():
        if len(features) <= sample_idx:
            continue
            
        feature_map = features[sample_idx]  # Shape: (H, W, D, C)
        
        # Resize void mask to match feature map spatial dimensions
        target_shape = feature_map.shape[:3]
        if void_mask.shape != target_shape:
            scale_factors = [target_shape[i] / void_mask.shape[i] for i in range(3)]
            resized_void_mask = zoom(void_mask, scale_factors, order=0)
        else:
            resized_void_mask = void_mask
        
        # Calculate correlations for each filter
        filter_correlations = []
        for filter_idx in range(feature_map.shape[-1]):
            activation = feature_map[:, :, :, filter_idx]
            try:
                correlation = np.corrcoef(activation.flatten(), resized_void_mask.flatten())[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            except:
                correlation = 0.0
            filter_correlations.append(correlation)
        
        correlations[layer_name] = np.array(filter_correlations)
    
    print(f'Calculated correlations for {len(correlations)} layers')
    return correlations

def plot_layer_evolution_analysis(feature_dict, void_mask, sample_idx=0, save_path=None):
    """
    Plot how feature representations evolve through network depth.
    """
    print('Creating layer evolution analysis plot...')
    
    if not feature_dict:
        print("No feature data to plot")
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    layer_names = list(feature_dict.keys())
    layer_depths = np.arange(len(layer_names))
    
    # Calculate statistics for each layer
    layer_stats = {
        'sparsity': [],  # Fraction of near-zero activations
        'max_activation': [],  # Maximum activation value
        'mean_activation': [],  # Mean activation value
        'std_activation': [],  # Standard deviation
        'void_selectivity': [],  # Mean correlation with voids
        'feature_diversity': []  # Inter-filter correlation
    }
    
    for layer_name in layer_names:
        features = feature_dict[layer_name][sample_idx]  # Shape: (H, W, D, C)
        
        # Sparsity (fraction of activations < 0.01 * max)
        threshold = 0.01 * np.max(features)
        sparsity = np.mean(features < threshold)
        layer_stats['sparsity'].append(sparsity)
        
        # Activation statistics
        layer_stats['max_activation'].append(np.max(features))
        layer_stats['mean_activation'].append(np.mean(features))
        layer_stats['std_activation'].append(np.std(features))
        
        # Void selectivity (if correlations available)
        void_mask_resized = zoom(void_mask, 
                               [features.shape[i]/void_mask.shape[i] for i in range(3)], 
                               order=0)
        correlations = []
        for c in range(features.shape[-1]):
            corr = np.corrcoef(features[:,:,:,c].flatten(), void_mask_resized.flatten())[0,1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        layer_stats['void_selectivity'].append(np.mean(correlations) if correlations else 0)
        
        # Feature diversity (1 - mean inter-filter correlation)
        if features.shape[-1] > 1:
            flat_features = features.reshape(-1, features.shape[-1])
            filter_corr_matrix = np.corrcoef(flat_features.T)
            upper_tri = np.triu(filter_corr_matrix, k=1)
            mean_inter_corr = np.mean(upper_tri[upper_tri != 0])
            diversity = 1 - abs(mean_inter_corr) if not np.isnan(mean_inter_corr) else 0
        else:
            diversity = 0
        layer_stats['feature_diversity'].append(diversity)
    
    # Plot 1: Sparsity evolution
    ax = axes[0, 0]
    ax.plot(layer_depths, layer_stats['sparsity'], 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Sparsity (fraction < 1% max)')
    ax.set_title('Feature Sparsity Evolution')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Activation magnitude evolution
    ax = axes[0, 1]
    ax.semilogy(layer_depths, layer_stats['max_activation'], 'o-', label='Max', linewidth=2)
    ax.semilogy(layer_depths, layer_stats['mean_activation'], 's-', label='Mean', linewidth=2)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Activation Magnitude (log scale)')
    ax.set_title('Activation Magnitude Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Void selectivity evolution
    ax = axes[0, 2]
    ax.plot(layer_depths, layer_stats['void_selectivity'], 'o-', color='red', linewidth=2, markersize=6)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Mean |Correlation| with Voids')
    ax.set_title('Void Selectivity Evolution')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Feature diversity evolution
    ax = axes[1, 0]
    ax.plot(layer_depths, layer_stats['feature_diversity'], 'o-', color='green', linewidth=2, markersize=6)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Feature Diversity (1 - inter-correlation)')
    ax.set_title('Feature Diversity Evolution')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Combined normalized metrics
    ax = axes[1, 1]
    metrics_norm = {}
    for key, values in layer_stats.items():
        if key != 'max_activation':  # Skip log-scale metric
            norm_values = (np.array(values) - np.min(values)) / (np.max(values) - np.min(values) + 1e-8)
            metrics_norm[key] = norm_values
    
    for key, values in metrics_norm.items():
        ax.plot(layer_depths, values, 'o-', label=key.replace('_', ' ').title(), linewidth=2, markersize=4)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Normalized Metrics Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Activation distribution comparison (first vs last layer)
    ax = axes[1, 2]
    first_layer_features = feature_dict[layer_names[0]][sample_idx].flatten()
    last_layer_features = feature_dict[layer_names[-1]][sample_idx].flatten()
    
    ax.hist(first_layer_features, bins=50, alpha=0.6, label=f'First Layer ({layer_names[0][:15]})', density=True)
    ax.hist(last_layer_features, bins=50, alpha=0.6, label=f'Last Layer ({layer_names[-1][:15]})', density=True)
    ax.set_xlabel('Activation Value')
    ax.set_ylabel('Density')
    ax.set_title('Activation Distribution: First vs Last Layer')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f'Layer evolution analysis saved to {save_path}')
    
    return fig

def plot_class_specific_analysis(feature_dict, labels, sample_idx=0, save_path=None):
    """
    Analyze how different cosmic structure classes activate different filters.
    """
    print('Creating class-specific activation analysis plot...')
    
    if not feature_dict:
        print("No feature data to plot")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get class masks for the sample
    label_sample = labels[sample_idx, :, :, :, 0]  # Shape: (H, W, D)
    class_masks = {}
    class_names = ['void', 'wall', 'filament', 'halo']
    
    for class_idx, class_name in enumerate(class_names):
        class_masks[class_name] = (label_sample == class_idx).astype(float)
    
    # Select a representative layer (middle of network)
    layer_names = list(feature_dict.keys())
    mid_layer_idx = len(layer_names) // 2
    selected_layer = layer_names[mid_layer_idx]
    features = feature_dict[selected_layer][sample_idx]  # Shape: (H, W, D, C)
    
    print(f'Using layer: {selected_layer} with {features.shape[-1]} filters')
    
    # Calculate class-specific activations
    class_activations = {}
    for class_name, mask in class_masks.items():
        # Resize mask to match feature dimensions
        mask_resized = zoom(mask, [features.shape[i]/mask.shape[i] for i in range(3)], order=0)
        
        # Calculate mean activation for each filter in this class region
        activations = []
        for c in range(features.shape[-1]):
            feature_map = features[:, :, :, c]
            if np.sum(mask_resized) > 0:
                mean_activation = np.sum(feature_map * mask_resized) / np.sum(mask_resized)
            else:
                mean_activation = 0
            activations.append(mean_activation)
        class_activations[class_name] = np.array(activations)
    
    # Plot 1: Class-specific filter preferences
    ax = axes[0, 0]
    filter_indices = np.arange(min(32, features.shape[-1]))  # Show first 32 filters
    
    for class_name in class_names:
        activations = class_activations[class_name][filter_indices]
        ax.plot(filter_indices, activations, 'o-', label=class_name, linewidth=2, markersize=4)
    
    ax.set_xlabel('Filter Index')
    ax.set_ylabel('Mean Activation in Class Region')
    ax.set_title(f'Class-Specific Filter Activations\n({selected_layer})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Filter selectivity matrix
    ax = axes[0, 1]
    selectivity_matrix = np.array([class_activations[name] for name in class_names])
    
    # Normalize by row (each class)
    selectivity_norm = selectivity_matrix / (np.max(selectivity_matrix, axis=1, keepdims=True) + 1e-8)
    
    # Show only subset of filters for clarity
    n_filters_show = min(20, selectivity_norm.shape[1])
    im = ax.imshow(selectivity_norm[:, :n_filters_show], cmap='viridis', aspect='auto')
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Filter Index')
    ax.set_title('Normalized Filter Selectivity by Class')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Plot 3: Class discrimination analysis
    ax = axes[1, 0]
    
    # Calculate how well each filter discriminates between classes
    discrimination_scores = []
    for c in range(features.shape[-1]):
        activations = [class_activations[name][c] for name in class_names]
        # Use coefficient of variation as discrimination measure
        discrimination = np.std(activations) / (np.mean(activations) + 1e-8)
        discrimination_scores.append(discrimination)
    
    # Show top discriminative filters
    top_indices = np.argsort(discrimination_scores)[-20:]
    top_scores = np.array(discrimination_scores)[top_indices]
    
    bars = ax.bar(range(len(top_scores)), top_scores)
    ax.set_xlabel('Filter Rank (Top Discriminative)')
    ax.set_ylabel('Discrimination Score (CV)')
    ax.set_title('Most Class-Discriminative Filters')
    ax.grid(True, alpha=0.3)
    
    # Color bars by score
    max_score = max(top_scores) if len(top_scores) > 0 else 1
    for bar, score in zip(bars, top_scores):
        bar.set_color(plt.cm.plasma(score / max_score))
    
    # Plot 4: Class activation distributions
    ax = axes[1, 1]
    
    # Select top discriminative filter for detailed analysis
    if len(discrimination_scores) > 0:
        best_filter_idx = np.argmax(discrimination_scores)
        
        for class_name in class_names:
            mask = class_masks[class_name]
            mask_resized = zoom(mask, [features.shape[i]/mask.shape[i] for i in range(3)], order=0)
            feature_map = features[:, :, :, best_filter_idx]
            
            # Get activations in class regions
            class_activations_detailed = feature_map[mask_resized > 0.5]
            
            if len(class_activations_detailed) > 0:
                ax.hist(class_activations_detailed, bins=30, alpha=0.6, 
                       label=f'{class_name} (n={len(class_activations_detailed)})', density=True)
        
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Density')
        ax.set_title(f'Activation Distribution by Class\n(Filter {best_filter_idx}, highest discrimination)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f'Class-specific analysis saved to {save_path}')
    
    return fig

def plot_spatial_activation_patterns(feature_dict, original_input, void_mask, sample_idx=0, save_path=None):
    """
    Analyze spatial patterns of feature activations.
    """
    print('Creating spatial activation pattern analysis plot...')
    
    if not feature_dict:
        print("No feature data to plot")
        return None
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Select layers at different depths
    layer_names = list(feature_dict.keys())
    if len(layer_names) >= 3:
        selected_layers = [layer_names[0], layer_names[len(layer_names)//2], layer_names[-1]]
        layer_labels = ['Early', 'Middle', 'Late']
    else:
        selected_layers = layer_names
        layer_labels = [f'Layer {i+1}' for i in range(len(selected_layers))]
    
    # Get middle slice for visualization
    slice_idx = original_input.shape[2] // 2
    original_slice = original_input[sample_idx, :, :, slice_idx, 0]
    void_slice = void_mask[:, :, slice_idx]
    
    for row, (layer_name, layer_label) in enumerate(zip(selected_layers, layer_labels)):
        features = feature_dict[layer_name][sample_idx]  # Shape: (H, W, D, C)
        
        # Get middle slice of features
        if len(features.shape) == 4:
            feature_slice_idx = features.shape[2] // 2
            feature_slice = features[:, :, feature_slice_idx, :]  # Shape: (H, W, C)
        else:
            feature_slice = features
        
        # Resize to match original if needed
        if feature_slice.shape[:2] != original_slice.shape:
            target_shape = original_slice.shape
            feature_slice_resized = np.zeros(target_shape + (feature_slice.shape[-1],))
            for c in range(feature_slice.shape[-1]):
                feature_slice_resized[:, :, c] = zoom(feature_slice[:, :, c], 
                                                    [target_shape[i]/feature_slice.shape[i] for i in range(2)], 
                                                    order=1)
            feature_slice = feature_slice_resized
        
        # Column 1: Original input with void overlay
        ax = axes[row, 0]
        ax.imshow(original_slice, cmap='gray', alpha=0.8)
        ax.contour(void_slice, levels=[0.5], colors='red', alpha=0.6, linewidths=1)
        ax.set_title(f'{layer_label}: Input + Void Contours')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Column 2: Mean activation map
        ax = axes[row, 1]
        mean_activation = np.mean(feature_slice, axis=-1)
        im = ax.imshow(mean_activation, cmap='hot')
        ax.contour(void_slice, levels=[0.5], colors='cyan', alpha=0.8, linewidths=1)
        ax.set_title(f'{layer_label}: Mean Activation')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Column 3: Activation standard deviation (feature diversity)
        ax = axes[row, 2]
        std_activation = np.std(feature_slice, axis=-1)
        im = ax.imshow(std_activation, cmap='viridis')
        ax.contour(void_slice, levels=[0.5], colors='white', alpha=0.8, linewidths=1)
        ax.set_title(f'{layer_label}: Activation Diversity')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f'Spatial activation patterns saved to {save_path}')
    
    return fig

def plot_void_correlation_analysis(correlations, save_path=None):
    """
    Plot analysis of void-activation correlations.
    """
    print('Creating void correlation analysis plot...')
    
    if not correlations:
        print("No correlation data to plot")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Top-left: Layer-wise void selectivity
    ax = axes[0, 0]
    layer_selectivity = []
    layer_names_short = []
    for layer_name, corrs in correlations.items():
        avg_selectivity = np.mean(np.abs(corrs))
        layer_selectivity.append(avg_selectivity)
        # Shorten layer names for display
        short_name = layer_name.replace('encoder_block_', 'E').replace('decoder_block_', 'D')
        short_name = short_name.replace('bottleneck', 'BN')[:15]
        layer_names_short.append(short_name)
    
    bars = ax.bar(range(len(layer_selectivity)), layer_selectivity)
    ax.set_xticks(range(len(layer_names_short)))
    ax.set_xticklabels(layer_names_short, rotation=45, ha='right')
    ax.set_ylabel('Average |Correlation| with Voids')
    ax.set_title('Void Selectivity by Layer')
    ax.grid(True, alpha=0.3)
    
    # Color bars by selectivity
    max_sel = max(layer_selectivity) if layer_selectivity else 1
    for bar, sel in zip(bars, layer_selectivity):
        bar.set_color(plt.cm.viridis(sel / max_sel))
    
    # Top-right: Distribution of correlations
    ax = axes[0, 1]
    all_correlations = []
    for corrs in correlations.values():
        all_correlations.extend(corrs)
    
    if all_correlations:
        ax.hist(all_correlations, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='No correlation')
        ax.set_xlabel('Correlation with Voids')
        ax.set_ylabel('Number of Filters')
        ax.set_title('Distribution of Void Correlations')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Bottom-left: Top void-selective filters
    ax = axes[1, 0]
    all_corr_data = []
    for layer_name, corrs in correlations.items():
        for filter_idx, corr in enumerate(corrs):
            all_corr_data.append((abs(corr), corr, layer_name, filter_idx))
    
    # Sort by absolute correlation and take top 15
    all_corr_data.sort(reverse=True)
    top_data = all_corr_data[:15]
    
    if top_data:
        top_corrs = [data[1] for data in top_data]  # Use signed correlation
        top_labels = [f"{data[2][:10]}_{data[3]}" for data in top_data]
        
        colors = ['red' if corr < 0 else 'blue' for corr in top_corrs]
        bars = ax.barh(range(len(top_corrs)), top_corrs, color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_labels)))
        ax.set_yticklabels(top_labels, fontsize=8)
        ax.set_xlabel('Correlation with Voids')
        ax.set_title('Top Void-Selective Filters')
        ax.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    # Bottom-right: Correlation vs layer depth
    ax = axes[1, 1]
    layer_depths = []
    layer_mean_corrs = []
    layer_std_corrs = []
    
    for i, (layer_name, corrs) in enumerate(correlations.items()):
        layer_depths.append(i)
        layer_mean_corrs.append(np.mean(corrs))
        layer_std_corrs.append(np.std(corrs))
    
    if layer_depths:
        ax.errorbar(layer_depths, layer_mean_corrs, yerr=layer_std_corrs, 
                   marker='o', capsize=5, capthick=2)
        ax.set_xlabel('Layer Index (Network Depth)')
        ax.set_ylabel('Mean Correlation with Voids')
        ax.set_title('Void Selectivity vs Network Depth')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f'Void correlation analysis saved to {save_path}')
    
    return fig

#================================================================
# Data loading function
#================================================================
def load_analysis_data(inter_sep, max_samples=MAX_SAMPLES):
    """
    Load data for interpretability analysis.
    """
    print(f'Loading analysis data for L={inter_sep} Mpc/h...')
    
    if inter_sep not in data_info:
        raise ValueError(f'Invalid interparticle separation: {inter_sep}')
    
    data_file = DATA_PATH + data_info[inter_sep]
    
    # Check if files exist
    if not os.path.exists(data_file):
        raise FileNotFoundError(f'Data file not found: {data_file}')
    if not os.path.exists(FILE_MASK):
        raise FileNotFoundError(f'Mask file not found: {FILE_MASK}')
    
    # Map preprocessing options
    preproc_mapping = {
        'standard': 'mm',
        'robust': 'robust',
        'log_transform': 'log_transform',
        'clip_extreme': 'clip_extreme'
    }
    preproc_param = preproc_mapping.get(PREPROCESSING, 'mm')
    
    try:
        # Load features using appropriate method based on training configuration
        print('Loading density features...')
        
        if USE_OVERLAPPING_SUBCUBES:
            # Use overlapping subcubes for more diverse samples
            if RSD_PRESERVING_ROTATIONS:
                if EXTRA_AUGMENTATION:
                    features = nets.load_dataset_all_overlap_rsd_preserving(
                        data_file, FILE_MASK, SUBGRID, OFF, preproc=preproc_param, preprocessing=PREPROCESSING
                    )[0]  # Only get features, not labels
                else:
                    features = nets.load_dataset_all_overlap_rsd_preserving_light(
                        data_file, FILE_MASK, SUBGRID, OFF, preproc=preproc_param, preprocessing=PREPROCESSING
                    )[0]  # Only get features, not labels
            else:
                if EXTRA_AUGMENTATION:
                    features = nets.load_dataset_all_overlap(
                        data_file, FILE_MASK, SUBGRID, OFF, preproc=preproc_param, preprocessing=PREPROCESSING
                    )[0]  # Only get features, not labels
                else:
                    features = nets.load_dataset_all_overlap_light(
                        data_file, FILE_MASK, SUBGRID, OFF, preproc=preproc_param, preprocessing=PREPROCESSING
                    )[0]  # Only get features, not labels
        else:
            # Use non-overlapping subcubes (memory efficient)
            if RSD_PRESERVING_ROTATIONS:
                features = nets.load_dataset_all_rsd_preserving_light(
                    data_file, FILE_MASK, SUBGRID, preproc=preproc_param, preprocessing=PREPROCESSING
                )[0]  # Only get features, not labels
            else:
                features = nets.load_dataset_all_light(
                    data_file, FILE_MASK, SUBGRID, preproc=preproc_param, preprocessing=PREPROCESSING
                )[0]  # Only get features, not labels
        
        # Load labels using the same method
        print('Loading structure labels...')
        
        if USE_OVERLAPPING_SUBCUBES:
            if RSD_PRESERVING_ROTATIONS:
                if EXTRA_AUGMENTATION:
                    labels = nets.load_dataset_all_overlap_rsd_preserving(
                        data_file, FILE_MASK, SUBGRID, OFF, preproc=None, preprocessing=None
                    )[1]  # Only get labels, not features
                else:
                    labels = nets.load_dataset_all_overlap_rsd_preserving_light(
                        data_file, FILE_MASK, SUBGRID, OFF, preproc=None, preprocessing=None
                    )[1]  # Only get labels, not features
            else:
                if EXTRA_AUGMENTATION:
                    labels = nets.load_dataset_all_overlap(
                        data_file, FILE_MASK, SUBGRID, OFF, preproc=None, preprocessing=None
                    )[1]  # Only get labels, not features
                else:
                    labels = nets.load_dataset_all_overlap_light(
                        data_file, FILE_MASK, SUBGRID, OFF, preproc=None, preprocessing=None
                    )[1]  # Only get labels, not features
        else:
            if RSD_PRESERVING_ROTATIONS:
                labels = nets.load_dataset_all_rsd_preserving_light(
                    data_file, FILE_MASK, SUBGRID, preproc=None, preprocessing=None
                )[1]  # Only get labels, not features
            else:
                labels = nets.load_dataset_all_light(
                    data_file, FILE_MASK, SUBGRID, preproc=None, preprocessing=None
                )[1]  # Only get labels, not features
        
        labels = labels.astype(np.int32)
        
        print(f'Loaded features: {features.shape}, labels: {labels.shape}')
        
        # Load extra inputs if specified
        if EXTRA_INPUTS is not None:
            print(f'Loading extra inputs: {EXTRA_INPUTS}...')
            extra_input_file = DATA_PATH + EXTRA_INPUTS_INFO[inter_sep]
            
            if not os.path.exists(extra_input_file):
                raise FileNotFoundError(f'Extra input file not found: {extra_input_file}')
            
            if USE_OVERLAPPING_SUBCUBES:
                if RSD_PRESERVING_ROTATIONS:
                    if EXTRA_AUGMENTATION:
                        extra_features = nets.load_dataset_all_overlap_rsd_preserving(
                            extra_input_file, FILE_MASK, SUBGRID, OFF, preproc=preproc_param, preprocessing=PREPROCESSING
                        )[0]  # Only get features, not labels
                    else:
                        extra_features = nets.load_dataset_all_overlap_rsd_preserving_light(
                            extra_input_file, FILE_MASK, SUBGRID, OFF, preproc=preproc_param, preprocessing=PREPROCESSING
                        )[0]  # Only get features, not labels
                else:
                    if EXTRA_AUGMENTATION:
                        extra_features = nets.load_dataset_all_overlap(
                            extra_input_file, FILE_MASK, SUBGRID, OFF, preproc=preproc_param, preprocessing=PREPROCESSING
                        )[0]  # Only get features, not labels
                    else:
                        extra_features = nets.load_dataset_all_overlap_light(
                            extra_input_file, FILE_MASK, SUBGRID, OFF, preproc=preproc_param, preprocessing=PREPROCESSING
                        )[0]  # Only get features, not labels
            else:
                if RSD_PRESERVING_ROTATIONS:
                    extra_features = nets.load_dataset_all_rsd_preserving_light(
                        extra_input_file, FILE_MASK, SUBGRID, preproc=preproc_param, preprocessing=PREPROCESSING
                    )[0]  # Only get features, not labels
                else:
                    extra_features = nets.load_dataset_all_light(
                        extra_input_file, FILE_MASK, SUBGRID, preproc=preproc_param, preprocessing=PREPROCESSING
                    )[0]  # Only get features, not labels
            
            print(f'Extra inputs loaded: {extra_features.shape}')
            features = np.concatenate([features, extra_features], axis=-1)
            print(f'Combined features: {features.shape}')
            del extra_features
            gc.collect()
        
        # Limit samples if requested
        if max_samples and features.shape[0] > max_samples:
            print(f'Limiting to first {max_samples} samples')
            features = features[:max_samples]
            labels = labels[:max_samples]
        
        print('Data loading completed successfully')
        return features, labels
        
    except Exception as e:
        print(f'Error loading data: {e}')
        raise

#================================================================
# Model loading function
#================================================================
def load_model_for_analysis():
    """
    Load the trained model for analysis.
    """
    print(f'Loading model: {MODEL_NAME}')
    
    # Try different model file formats
    model_paths = [
        MODEL_PATH + MODEL_NAME + f'_L{L_ANALYSIS}.keras',
        MODEL_PATH + MODEL_NAME + f'_L{L_ANALYSIS}.h5',
        MODEL_PATH + MODEL_NAME + '.keras',
        MODEL_PATH + MODEL_NAME + '.h5'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f'Attempting to load model from {model_path}...')
            try:
                model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
                print(f'Successfully loaded model from {model_path}')
                return model
            except Exception as e:
                print(f'Failed to load model from {model_path}: {e}')
                continue
    
    raise FileNotFoundError(f'Could not find or load model: {MODEL_NAME}')

#================================================================
# Main analysis function
#================================================================
def run_interpretability_analysis():
    """
    Run comprehensive interpretability analysis.
    """
    global EXTRA_INPUTS, EXTRA_INPUTS_INFO  # Allow modification of global variables
    
    print('=' * 80)
    print('DEEPVOID MODEL INTERPRETABILITY ANALYSIS')
    print('=' * 80)
    print(f'Model: {MODEL_NAME}')
    print(f'Analysis Lambda: {L_ANALYSIS} Mpc/h')
    print(f'Max Samples: {MAX_SAMPLES}')
    print(f'Results Directory: {ANALYSIS_PATH}')
    print('=' * 80)
    
    # Load model
    try:
        model = load_model_for_analysis()
        print(f'Model loaded successfully')
        print(f'Model has {len(model.layers)} layers')
    except Exception as e:
        print(f'Error loading model: {e}')
        return
    
    # Load data
    try:
        features, labels = load_analysis_data(L_ANALYSIS, MAX_SAMPLES)
        print(f'Data loaded: {features.shape} features, {labels.shape} labels')
        
        # Check if model expects more input channels than we loaded
        expected_input_shape = model.input_shape
        print(f'Model expects input shape: {expected_input_shape}')
        print(f'Loaded data shape: {features.shape}')
        
        if expected_input_shape[-1] != features.shape[-1]:
            print(f'WARNING: Model expects {expected_input_shape[-1]} input channels but data has {features.shape[-1]} channels')
            
            # Try to auto-detect missing extra inputs
            if expected_input_shape[-1] == 2 and features.shape[-1] == 1 and EXTRA_INPUTS is None:
                print('Attempting to auto-detect missing extra input...')
                # Check model name for hints
                if 'r_flux_density' in MODEL_NAME:
                    EXTRA_INPUTS = 'r_flux_density'
                    print('Auto-detected missing extra input: r_flux_density')
                elif 'g-r' in MODEL_NAME:
                    EXTRA_INPUTS = 'g-r' 
                    print('Auto-detected missing extra input: g-r')
                
                if EXTRA_INPUTS is not None:
                    print(f'Reloading data with extra input: {EXTRA_INPUTS}')
                    
                    # Update EXTRA_INPUTS_INFO if needed
                    if EXTRA_INPUTS == 'r_flux_density' and not EXTRA_INPUTS_INFO:
                        EXTRA_INPUTS_INFO = {
                            '0.33': 'subs1_r_flux_density_Nm512_L3.fvol',
                            '3': 'subs1_r_flux_density_Nm512_L3.fvol',
                            '5': 'subs1_r_flux_density_Nm512_L5.fvol',
                            '7': 'subs1_r_flux_density_Nm512_L7.fvol',
                            '10': 'subs1_r_flux_density_Nm512_L10.fvol',
                        }
                        if ADD_RSD:
                            for key in EXTRA_INPUTS_INFO:
                                EXTRA_INPUTS_INFO[key] = EXTRA_INPUTS_INFO[key].replace('.fvol', '_RSD.fvol')
                    elif EXTRA_INPUTS == 'g-r' and not EXTRA_INPUTS_INFO:
                        EXTRA_INPUTS_INFO = {
                            '0.33': 'subs1_g-r_Nm512_L3.fvol',
                            '3': 'subs1_g-r_Nm512_L3.fvol',
                            '5': 'subs1_g-r_Nm512_L5.fvol',
                            '7': 'subs1_g-r_Nm512_L7.fvol',
                            '10': 'subs1_g-r_Nm512_L10.fvol',
                        }
                        if ADD_RSD:
                            for key in EXTRA_INPUTS_INFO:
                                EXTRA_INPUTS_INFO[key] = EXTRA_INPUTS_INFO[key].replace('.fvol', '_RSD.fvol')
                    
                    try:
                        features, labels = load_analysis_data(L_ANALYSIS, MAX_SAMPLES)
                        print(f'Reloaded data: {features.shape} features, {labels.shape} labels')
                    except Exception as e:
                        print(f'Failed to reload with extra inputs: {e}')
                        print('Continuing with original data - model inference may fail')
                        
            if expected_input_shape[-1] != features.shape[-1]:
                print(f'ERROR: Cannot proceed - model expects {expected_input_shape[-1]} channels but data has {features.shape[-1]}')
                print('Please specify the correct --EXTRA_INPUTS parameter or check your model')
                return
                
    except Exception as e:
        print(f'Error loading data: {e}')
        return
    
    # Extract void mask from labels (assuming void = class 0)
    void_mask = (labels[0, :, :, :, 0] == 0).astype(float)
    print(f'Void mask shape: {void_mask.shape}')
    print(f'Void fraction: {np.mean(void_mask):.3f}')
    
    # Get slice index
    slice_idx_use = SLICE_IDX if SLICE_IDX is not None else void_mask.shape[2] // 2
    print(f'Using slice index: {slice_idx_use}')
    
    # 1. Extract feature activation maps
    print('\n--- Extracting Feature Activation Maps ---')
    feature_dict = extract_feature_maps(model, features, max_samples=MAX_SAMPLES)
    
    if feature_dict and SAVE_ALL:
        print('Creating feature maps visualization...')
        fig1 = plot_feature_maps_3d(
            feature_dict, 
            sample_idx=0, 
            slice_idx=slice_idx_use,
            max_filters=MAX_FILTERS,
            void_regions=void_mask[:, :, slice_idx_use],
            save_path=os.path.join(ANALYSIS_PATH, 'feature_activation_maps.png')
        )
        plt.close(fig1)
    
    # 2. Extract attention maps (if attention model)
    if USE_ATTENTION:
        print('\n--- Extracting Attention Maps ---')
        attention_dict = extract_attention_maps(model, features, max_samples=MAX_SAMPLES)
        
        if attention_dict and SAVE_ALL:
            print('Creating attention maps visualization...')
            fig2 = plot_attention_maps_3d(
                attention_dict,
                features,
                void_mask=void_mask,
                sample_idx=0,
                slice_idx=slice_idx_use,
                save_path=os.path.join(ANALYSIS_PATH, 'attention_maps.png')
            )
            plt.close(fig2)
    else:
        print('\n--- Skipping Attention Analysis (Standard U-Net) ---')
    
    # 3. Analyze void-activation correlations
    print('\n--- Analyzing Void-Activation Correlations ---')
    if feature_dict:
        correlations = analyze_void_activation_correlation(feature_dict, void_mask, sample_idx=0)
        
        if correlations and SAVE_ALL:
            print('Creating void correlation analysis...')
            fig3 = plot_void_correlation_analysis(
                correlations,
                save_path=os.path.join(ANALYSIS_PATH, 'void_correlation_analysis.png')
            )
            plt.close(fig3)
            
            # Save correlation data
            correlation_file = os.path.join(ANALYSIS_PATH, 'void_correlations.npz')
            np.savez(correlation_file, **correlations)
            print(f'Correlation data saved to {correlation_file}')
    
    # 4. Layer evolution analysis
    print('\n--- Analyzing Layer Evolution ---')
    if feature_dict and SAVE_ALL:
        print('Creating layer evolution analysis...')
        fig4 = plot_layer_evolution_analysis(
            feature_dict, 
            void_mask, 
            sample_idx=0,
            save_path=os.path.join(ANALYSIS_PATH, 'layer_evolution_analysis.png')
        )
        plt.close(fig4)
    
    # 5. Class-specific analysis
    print('\n--- Analyzing Class-Specific Activations ---')
    if feature_dict and SAVE_ALL:
        print('Creating class-specific analysis...')
        fig5 = plot_class_specific_analysis(
            feature_dict, 
            labels, 
            sample_idx=0,
            save_path=os.path.join(ANALYSIS_PATH, 'class_specific_analysis.png')
        )
        plt.close(fig5)
    
    # 6. Spatial activation patterns
    print('\n--- Analyzing Spatial Activation Patterns ---')
    if feature_dict and SAVE_ALL:
        print('Creating spatial activation patterns...')
        fig6 = plot_spatial_activation_patterns(
            feature_dict, 
            features, 
            void_mask, 
            sample_idx=0,
            save_path=os.path.join(ANALYSIS_PATH, 'spatial_activation_patterns.png')
        )
        plt.close(fig6)
    
    # 7. Generate summary report
    print('\n--- Generating Analysis Summary ---')
    summary_file = os.path.join(ANALYSIS_PATH, 'analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"DeepVoid Model Interpretability Analysis\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Analysis Lambda: {L_ANALYSIS} Mpc/h\n")
        f.write(f"Preprocessing: {PREPROCESSING}\n")
        f.write(f"Extra Inputs: {EXTRA_INPUTS}\n")
        f.write(f"RSD: {ADD_RSD}\n")
        f.write(f"Attention: {USE_ATTENTION}\n")
        f.write(f"Data Loading Method:\n")
        f.write(f"- Overlapping subcubes: {USE_OVERLAPPING_SUBCUBES}\n")
        f.write(f"- RSD-preserving rotations: {RSD_PRESERVING_ROTATIONS}\n")
        f.write(f"- Extra augmentation: {EXTRA_AUGMENTATION}\n")
        f.write(f"Samples Analyzed: {MAX_SAMPLES}\n")
        f.write(f"Slice Index: {slice_idx_use}\n\n")
        
        f.write(f"Data Summary:\n")
        f.write(f"- Features shape: {features.shape}\n")
        f.write(f"- Labels shape: {labels.shape}\n")
        f.write(f"- Void fraction: {np.mean(void_mask):.3f}\n\n")
        
        f.write(f"Model Summary:\n")
        f.write(f"- Total layers: {len(model.layers)}\n")
        f.write(f"- Feature layers analyzed: {len(feature_dict)}\n")
        if USE_ATTENTION:
            f.write(f"- Attention layers found: {len(attention_dict) if 'attention_dict' in locals() else 0}\n")
        f.write(f"\n")
        
        if 'correlations' in locals() and correlations:
            f.write(f"Void Correlation Analysis:\n")
            for layer_name, corrs in correlations.items():
                mean_corr = np.mean(corrs)
                max_corr = np.max(np.abs(corrs))
                f.write(f"- {layer_name}: mean={mean_corr:.3f}, max_abs={max_corr:.3f}\n")
        
        f.write(f"\nFiles Generated:\n")
        if SAVE_ALL:
            f.write(f"- feature_activation_maps.png\n")
            if USE_ATTENTION:
                f.write(f"- attention_maps.png\n")
            f.write(f"- void_correlation_analysis.png\n")
            f.write(f"- layer_evolution_analysis.png\n")
            f.write(f"- class_specific_analysis.png\n")
            f.write(f"- spatial_activation_patterns.png\n")
            f.write(f"- void_correlations.npz\n")
        f.write(f"- analysis_summary.txt\n")
        f.write(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f'Analysis summary saved to {summary_file}')
    
    # Clean up memory
    del features, labels, model
    if 'feature_dict' in locals():
        del feature_dict
    if 'attention_dict' in locals():
        del attention_dict
    gc.collect()
    
    print('\n' + '=' * 80)
    print('INTERPRETABILITY ANALYSIS COMPLETE')
    print('=' * 80)
    print(f'Results saved to: {ANALYSIS_PATH}')
    print('=' * 80)

#================================================================
# Main execution
#================================================================
if __name__ == '__main__':
    try:
        run_interpretability_analysis()
    except Exception as e:
        print(f'Analysis failed with error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
