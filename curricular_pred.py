#!/usr/bin/env python3
'''
Prediction and evaluation script for curricular-trained models.
This script loads a trained curricular model and performs:
1. Predictions on validation/test data
2. Scoring and metrics calculation  
3. Slice plot generation
4. Memory-efficient batch processing for large datasets

Usage:
python curricular_pred.py ROOT_DIR MODEL_NAME L_PRED [options]
'''

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import datetime
import gc
from pathlib import Path
from tensorflow.keras import mixed_precision
import psutil

# Import your custom modules
sys.path.append('.')
import NETS_LITE as nets
try:
    import plotter
except ImportError:
    print("Warning: plotter module not found. Slice plots will be skipped.")
    plotter = None

try:
    import volumes
except ImportError:
    print("Warning: volumes module not found. Data loading may fail.")
    volumes = None

print('>>> Running curricular_pred.py')
print('TensorFlow version:', tf.__version__)

# Set up GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
print('GPUs available:', gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Set random seed for reproducibility
seed = 12
print('Setting random seed:', seed)
np.random.seed(seed)
tf.random.set_seed(seed)

#================================================================
# Parse command line arguments
#================================================================
parser = argparse.ArgumentParser(description='Prediction script for curricular-trained models.')
required = parser.add_argument_group('required arguments')
required.add_argument('ROOT_DIR', type=str, help='Root directory for the project.')
required.add_argument('MODEL_NAME', type=str, help='Name of the trained model (without file extension).')
required.add_argument('L_PRED', type=str, help='Interparticle separation for prediction (e.g., "10").')

optional = parser.add_argument_group('optional arguments')
optional.add_argument('--BATCH_SIZE', type=int, default=2, 
                      help='Batch size for prediction. Default is 2 (very memory efficient).')
optional.add_argument('--EXTRA_INPUTS', type=str, default=None,
                      choices=['g-r', 'r_flux_density'],
                      help='Name of additional inputs used in training.')
optional.add_argument('--ADD_RSD', action='store_true',
                      help='Use RSD data files (same as training).')
optional.add_argument('--LAMBDA_CONDITIONING', action='store_true',
                      help='Model uses lambda conditioning.')
optional.add_argument('--SAVE_PREDICTIONS', action='store_true',
                      help='Save raw predictions to .fvol file.')
optional.add_argument('--SKIP_SLICE_PLOTS', action='store_true',
                      help='Skip slice plot generation to save memory.')
optional.add_argument('--MAX_PRED_BATCHES', type=int, default=None,
                      help='Limit number of prediction batches for memory management.')
optional.add_argument('--TEST_MODE', action='store_true',
                      help='Test mode: use only a small subset of data for quick testing.')

args = parser.parse_args()

ROOT_DIR = args.ROOT_DIR
MODEL_NAME = args.MODEL_NAME
L_PRED = args.L_PRED
BATCH_SIZE = args.BATCH_SIZE
EXTRA_INPUTS = args.EXTRA_INPUTS
ADD_RSD = args.ADD_RSD
LAMBDA_CONDITIONING = args.LAMBDA_CONDITIONING
SAVE_PREDICTIONS = args.SAVE_PREDICTIONS
SKIP_SLICE_PLOTS = args.SKIP_SLICE_PLOTS
MAX_PRED_BATCHES = args.MAX_PRED_BATCHES
TEST_MODE = args.TEST_MODE

print(f'Parsed arguments: ROOT_DIR={ROOT_DIR}, MODEL_NAME={MODEL_NAME}, L_PRED={L_PRED}')
print(f'BATCH_SIZE={BATCH_SIZE}, EXTRA_INPUTS={EXTRA_INPUTS}, ADD_RSD={ADD_RSD}')
print(f'LAMBDA_CONDITIONING={LAMBDA_CONDITIONING}, SAVE_PREDICTIONS={SAVE_PREDICTIONS}')

#================================================================
# Set up custom objects for model loading
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
# Set paths
#================================================================
DATA_PATH = ROOT_DIR + 'data/TNG/'
FIG_PATH = ROOT_DIR + 'figs/TNG/'
MODEL_PATH = ROOT_DIR + 'models/'
PRED_PATH = ROOT_DIR + 'preds/'

# Ensure output directories exist
os.makedirs(FIG_PATH, exist_ok=True)
os.makedirs(PRED_PATH, exist_ok=True)

#================================================================
# Set parameters
#================================================================
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

inter_seps = list(data_info.keys())

if L_PRED not in inter_seps:
    raise ValueError(f'Invalid interparticle separation: {L_PRED}. Must be one of {inter_seps}.')

#================================================================
# Memory monitoring function
#================================================================
def check_memory_usage():
    """Check current memory usage."""
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f'Current memory usage: {memory_mb:.1f} MB')
        return memory_mb
    except ImportError:
        # psutil not available
        return None

#================================================================
# Memory-efficient data loading function
#================================================================
def load_data_for_prediction(inter_sep, extra_inputs=None, max_samples=None):
    """
    Load data for prediction with memory management.
    """
    if inter_sep not in inter_seps:
        raise ValueError(f'Invalid interparticle separation: {inter_sep}. Must be one of {inter_seps}.')
    
    data_file = DATA_PATH + data_info[inter_sep]
    print(f'Loading prediction data from {data_file}...')
    
    # Check if file exists
    if not os.path.exists(data_file):
        raise FileNotFoundError(f'Data file not found: {data_file}')
    if not os.path.exists(FILE_MASK):
        raise FileNotFoundError(f'Mask file not found: {FILE_MASK}')
    
    print('Files exist, starting data loading...')
    
    try:
        features, labels = nets.load_dataset_all(
            FILE_DEN=data_file,
            FILE_MASK=FILE_MASK,
            SUBGRID=SUBGRID
        )
        print(f'Data loading completed successfully.')
    except Exception as e:
        print(f'Error during data loading: {e}')
        raise
    
    if max_samples and features.shape[0] > max_samples:
        print(f'Limiting to {max_samples} samples for memory management...')
        features = features[:max_samples]
        labels = labels[:max_samples]
    
    if extra_inputs is not None:
        if inter_sep not in EXTRA_INPUTS_INFO:
            raise ValueError(f'Invalid interparticle separation for extra inputs: {inter_sep}.')
        extra_input_file = DATA_PATH + EXTRA_INPUTS_INFO[inter_sep]
        print(f'Loading extra inputs from {extra_input_file}...')
        # Load extra inputs (implementation depends on your data structure)
        # extra_data = load_extra_inputs(extra_input_file)
        # features = np.concatenate([features, extra_data], axis=-1)
    
    print(f'Features shape: {features.shape}, Labels shape: {labels.shape}')
    return features, labels

def make_prediction_dataset(delta, tij_labels, batch_size=BATCH_SIZE, lambda_value=None):
    """
    Create a TensorFlow dataset for prediction.
    """
    print(f'Creating prediction dataset with {len(delta)} samples...')
    
    # Ensure correct data types
    delta = tf.convert_to_tensor(delta, dtype=tf.float32)
    tij_labels = tf.convert_to_tensor(tij_labels, dtype=tf.float32)
    
    if lambda_value is not None:
        print(f'Adding lambda input with value {lambda_value}')
        lambda_input = tf.fill([tf.shape(delta)[0], 1], lambda_value)
        dataset = tf.data.Dataset.from_tensor_slices((delta, lambda_input, tij_labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((delta, tij_labels))
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print(f'Prediction dataset created with {len(dataset)} batches.')
    return dataset

#================================================================
# Model loading with robust fallback
#================================================================
def load_curricular_model(model_name, l_pred):
    """
    Load a curricular model with fallback options.
    """
    model = None
    
    # Try different model file formats and locations
    model_paths = [
        MODEL_PATH + model_name + f'_L{l_pred}.keras',
        MODEL_PATH + model_name + f'_L{l_pred}.h5',
        MODEL_PATH + model_name + '.keras',
        MODEL_PATH + model_name + '.h5'
    ]
    
    weights_paths = [
        MODEL_PATH + model_name + f'_L{l_pred}_final.weights.h5',
        MODEL_PATH + model_name + f'_L{l_pred}_weights.weights.h5',
        MODEL_PATH + model_name + f'_L{l_pred}_final_weights.h5',
        MODEL_PATH + model_name + f'_L{l_pred}_weights.h5'
    ]
    
    # Try to load full model first
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f'Attempting to load full model from {model_path}...')
            try:
                model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
                print(f'Successfully loaded full model from {model_path}')
                return model
            except Exception as e:
                print(f'Failed to load full model: {e}')
                continue
    
    # If full model loading fails, try to recreate model and load weights
    print('Full model loading failed. Attempting to recreate model and load weights...')
    
    # Parse model name to extract architecture details
    try:
        model_info = parse_model_name_for_recreation(model_name)
        print(f'Parsed model info: {model_info}')
        
        # Recreate model architecture
        model = recreate_model_architecture(model_info)
        
        if model is not None:
            # Try to load weights
            for weights_path in weights_paths:
                if os.path.exists(weights_path):
                    print(f'Attempting to load weights from {weights_path}...')
                    try:
                        model.load_weights(weights_path)
                        print(f'Successfully loaded weights from {weights_path}')
                        return model
                    except Exception as e:
                        print(f'Failed to load weights from {weights_path}: {e}')
                        continue
        
    except Exception as e:
        print(f'Model recreation failed: {e}')
    
    print('Available weight files:')
    for weights_path in weights_paths:
        if os.path.exists(weights_path):
            print(f'  - {weights_path}')
    
    return None

def parse_model_name_for_recreation(model_name):
    """
    Parse model name to extract architecture parameters.
    Example: TNG_curricular_SCCE_Class_Penalty_D3_F8_RSD_attention_2025-08-05_21-27-59
    """
    parts = model_name.split('_')
    
    model_info = {
        'depth': 3,  # default
        'filters': 32,  # default
        'use_attention': False,
        'add_rsd': False,
        'lambda_conditioning': LAMBDA_CONDITIONING,
        'extra_inputs': EXTRA_INPUTS,
        'loss': 'SCCE'
    }
    
    for i, part in enumerate(parts):
        if part.startswith('D') and len(part) > 1:
            try:
                model_info['depth'] = int(part[1:])
            except ValueError:
                pass
        elif part.startswith('F') and len(part) > 1:
            try:
                model_info['filters'] = int(part[1:])
            except ValueError:
                pass
        elif part == 'attention':
            model_info['use_attention'] = True
        elif part == 'RSD':
            model_info['add_rsd'] = True
        elif part in ['SCCE', 'CCE', 'FOCAL_CCE', 'SCCE_Class_Penalty', 'SCCE_Void_Penalty']:
            model_info['loss'] = part
    
    return model_info

def recreate_model_architecture(model_info):
    """
    Recreate the model architecture based on parsed information.
    """
    try:
        # Determine input shape
        input_shape = (None, None, None, 1)
        if model_info['extra_inputs']:
            input_shape = (None, None, None, 2)  # Add channel for extra inputs
        
        print(f"Recreating model with input_shape={input_shape}")
        print(f"Depth={model_info['depth']}, Filters={model_info['filters']}")
        print(f"Attention={model_info['use_attention']}, Lambda conditioning={model_info['lambda_conditioning']}")
        
        # Create model based on architecture
        if model_info['use_attention']:
            model = nets.attention_unet_3d(
                input_shape=input_shape,
                num_classes=N_CLASSES,
                initial_filters=model_info['filters'],
                depth=model_info['depth'],
                activation='relu',
                last_activation='softmax',
                batch_normalization=True,
                BN_scheme='last',
                dropout_rate=None,
                DROP_scheme='last',
                REG_FLAG=False,
                model_name='Recreated_Attention_3D_U_Net',
                report_params=False,
                lambda_conditioning=model_info['lambda_conditioning']
            )
        else:
            model = nets.unet_3d(
                input_shape=input_shape,
                num_classes=N_CLASSES,
                initial_filters=model_info['filters'],
                depth=model_info['depth'],
                activation='relu',
                last_activation='softmax',
                batch_normalization=True,
                BN_scheme='last',
                dropout_rate=None,
                DROP_scheme='last',
                REG_FLAG=False,
                model_name='Recreated_3D_U_Net',
                report_params=False,
                lambda_conditioning=model_info['lambda_conditioning']
            )
        
        if model is not None:
            print('Model architecture recreated successfully')
            return model
        else:
            print('Failed to recreate model architecture')
            return None
            
    except Exception as e:
        print(f'Error recreating model architecture: {e}')
        return None

#================================================================
# Memory-efficient prediction function
#================================================================
def predict_with_memory_management(model, dataset, max_batches=None):
    """
    Make predictions with memory management.
    """
    predictions = []
    true_labels = []
    
    print('Making predictions...')
    batch_count = 0
    
    for batch_data in dataset:
        if max_batches and batch_count >= max_batches:
            print(f'Reached maximum batch limit ({max_batches})')
            break
            
        if len(batch_data) == 3:  # Lambda conditioning
            x_batch, lambda_batch, y_batch = batch_data
            batch_input = [x_batch, lambda_batch]
        else:  # No lambda conditioning
            x_batch, y_batch = batch_data
            batch_input = x_batch
        
        # Make prediction
        y_pred = model.predict(batch_input, verbose=0)
        
        # Handle different output formats
        if isinstance(y_pred, dict):
            y_pred_main = y_pred.get('last_activation', y_pred)
        elif isinstance(y_pred, list):
            y_pred_main = y_pred[0]
        else:
            y_pred_main = y_pred
        
        predictions.append(y_pred_main)
        true_labels.append(y_batch.numpy())
        
        batch_count += 1
        if batch_count % 10 == 0:
            print(f'Processed {batch_count} batches...')
        
        # Force garbage collection every few batches
        if batch_count % 20 == 0:
            gc.collect()
    
    # Concatenate all predictions
    if predictions:
        all_predictions = np.concatenate(predictions, axis=0)
        all_labels = np.concatenate(true_labels, axis=0)
        print(f'Predictions complete. Shape: {all_predictions.shape}')
        return all_predictions, all_labels
    else:
        return None, None

#================================================================
# Main execution
#================================================================
def main():
    print(f'Starting prediction for model: {MODEL_NAME}, L={L_PRED}')
    
    # Initial memory check
    initial_memory = check_memory_usage()
    if initial_memory and initial_memory > 8000:  # 8GB warning
        print(f'WARNING: High initial memory usage ({initial_memory:.1f} MB). Consider restarting kernel.')
    
    # Load the model
    model = load_curricular_model(MODEL_NAME, L_PRED)
    if model is None:
        print('ERROR: Could not load model. Please check model files.')
        return
    
    print('Model loaded successfully.')
    check_memory_usage()
    
    print(f'Model summary:')
    try:
        model.summary()
    except:
        print('Could not display model summary.')
    
    # Load prediction data
    try:
        # Aggressive memory management for Colab
        if TEST_MODE:
            max_samples = 16  # Even smaller for testing
            print('TEST_MODE: Using only 16 samples for quick testing')
        elif MAX_PRED_BATCHES:
            max_samples = MAX_PRED_BATCHES * BATCH_SIZE
            print(f'Limited to {max_samples} samples based on MAX_PRED_BATCHES={MAX_PRED_BATCHES}')
        else:
            # Default to a reasonable limit for Colab memory constraints
            max_samples = 64  # Conservative default for Colab
            print(f'Using default memory-safe limit of {max_samples} samples')
            
        pred_features, pred_labels = load_data_for_prediction(
            L_PRED, 
            extra_inputs=EXTRA_INPUTS,
            max_samples=max_samples
        )
        
        # Force garbage collection after data loading
        gc.collect()
        check_memory_usage()
        print(f'Memory cleanup completed after data loading')
        
    except Exception as e:
        print(f'ERROR: Failed to load prediction data: {e}')
        return
    
    # Create prediction dataset
    lambda_value = float(L_PRED) if LAMBDA_CONDITIONING else None
    pred_dataset = make_prediction_dataset(
        pred_features, pred_labels, 
        batch_size=BATCH_SIZE, 
        lambda_value=lambda_value
    )
    
    # Make predictions
    try:
        predictions, true_labels = predict_with_memory_management(
            model, pred_dataset, max_batches=MAX_PRED_BATCHES
        )
        
        if predictions is None:
            print('ERROR: Prediction failed.')
            return
            
    except Exception as e:
        print(f'ERROR: Prediction failed: {e}')
        return
    
    # Save predictions if requested
    if SAVE_PREDICTIONS:
        pred_file = PRED_PATH + MODEL_NAME + f'_predictions_L{L_PRED}.npy'
        try:
            np.save(pred_file, predictions)
            print(f'Predictions saved to {pred_file}')
        except Exception as e:
            print(f'Warning: Failed to save predictions: {e}')
    
    # Calculate scores
    print('Calculating scores...')
    scores = {}
    
    # Create output directory for figures
    MODEL_FIG_PATH = FIG_PATH + MODEL_NAME + '/'
    os.makedirs(MODEL_FIG_PATH, exist_ok=True)
    
    # Save model temporarily for scoring functions that require a model file
    temp_model_path = MODEL_PATH + MODEL_NAME + f'_temp_L{L_PRED}'
    print(f'Temporarily saving model for scoring functions...')
    
    try:
        # Save the recreated model temporarily (function will add .keras extension)
        model.save(temp_model_path + '.keras')
        print(f'Model temporarily saved to {temp_model_path}.keras')
        
        # Use your existing save_scores_from_fvol function (includes MCC)
        print('Using save_scores_from_fvol for comprehensive metrics (including MCC)...')
        nets.save_scores_from_fvol(
            true_labels, predictions, 
            temp_model_path,
            MODEL_FIG_PATH, scores, N_CLASSES, VAL_FLAG=True
        )
        print('save_scores_from_fvol completed successfully.')
        
        # Print key metrics from your scoring function
        if scores:
            print('\n=== Key Metrics (from save_scores_from_fvol) ===')
            for key, value in scores.items():
                if isinstance(value, (int, float)):
                    print(f'{key}: {value:.4f}')
        
    except Exception as e:
        print(f'save_scores_from_fvol failed: {e}')
        print('Falling back to direct sklearn scoring...')
        
        # Fallback to direct sklearn scoring
        try:
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from sklearn.metrics import f1_score, precision_score, recall_score
            
            # Convert predictions to class labels
            if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
                pred_labels = np.argmax(predictions, axis=-1)
            else:
                pred_labels = predictions
                
            # Convert true labels to class labels if needed  
            if len(true_labels.shape) > 1 and true_labels.shape[-1] > 1:
                true_labels_flat = np.argmax(true_labels, axis=-1)
            else:
                true_labels_flat = true_labels
                
            # Flatten for sklearn metrics
            y_true_flat = true_labels_flat.flatten()
            y_pred_flat = pred_labels.flatten()
            
            # Calculate comprehensive metrics
            accuracy = accuracy_score(y_true_flat, y_pred_flat)
            f1_macro = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
            f1_micro = f1_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
            f1_weighted = f1_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
            
            # Per-class F1 scores
            f1_per_class = f1_score(y_true_flat, y_pred_flat, average=None, zero_division=0)
            
            # Store in scores dict
            scores['accuracy'] = accuracy
            scores['f1_macro'] = f1_macro
            scores['f1_micro'] = f1_micro
            scores['f1_weighted'] = f1_weighted
            for i, class_name in enumerate(class_labels):
                if i < len(f1_per_class):
                    scores[f'f1_{class_name}'] = f1_per_class[i]
            
            print('Fallback scoring completed successfully.')
            
            # Print key metrics
            print('\n=== Key Metrics (fallback) ===')
            print(f'Accuracy: {accuracy:.4f}')
            print(f'F1 Macro: {f1_macro:.4f}')
            print(f'F1 Micro: {f1_micro:.4f}')
            print(f'F1 Weighted: {f1_weighted:.4f}')
            
            # Per-class F1 scores
            print('\n=== Per-Class F1 Scores ===')
            for i, class_name in enumerate(class_labels):
                if i < len(f1_per_class):
                    print(f'F1 {class_name}: {f1_per_class[i]:.4f}')
            
            # Classification report
            print('\n=== Classification Report ===')
            print(classification_report(y_true_flat, y_pred_flat, 
                                       target_names=class_labels, zero_division=0))
            
            # Save confusion matrix
            try:
                cm = confusion_matrix(y_true_flat, y_pred_flat)
                cm_file = MODEL_FIG_PATH + f'{MODEL_NAME}_confusion_matrix_L{L_PRED}.npy'
                np.save(cm_file, cm)
                print(f'Confusion matrix saved to {cm_file}')
            except Exception as e:
                print(f'Warning: Could not save confusion matrix: {e}')
                                           
        except Exception as e2:
            print(f'Fallback scoring also failed: {e2}')
            print('Continuing without detailed metrics...')
    
    # Generate slice plots (if not skipped and plotter is available)
    if not SKIP_SLICE_PLOTS:
        if plotter is not None:
            print('Generating slice plots using save_scores_from_model...')
            try:
                # Use your existing save_scores_from_model function with the temporary model
                FILE_PRED = PRED_PATH + MODEL_NAME + f'_predictions_L{L_PRED}.fvol'
                
                nets.save_scores_from_model(
                    DATA_PATH + data_info[L_PRED],  # FILE_DEN
                    FILE_MASK,                      # FILE_MSK  
                    temp_model_path,                # FILE_MODEL
                    MODEL_FIG_PATH,                 # FILE_FIG
                    FILE_PRED,                      # FILE_PRED
                    GRID=GRID, 
                    SUBGRID=SUBGRID, 
                    OFF=OFF,
                    TRAIN_SCORE=False,
                    EXTRA_INPUTS=EXTRA_INPUTS_INFO.get(L_PRED) if EXTRA_INPUTS and EXTRA_INPUTS_INFO else None,
                    lambda_value=float(L_PRED) if LAMBDA_CONDITIONING else None
                )
                print('Slice plots generated successfully using save_scores_from_model.')
                
            except Exception as e:
                print(f'Slice plot generation failed: {e}')
                print('This may be due to model serialization or data loading issues.')
        else:
            print('Slice plots skipped (plotter module not available).')
    else:
        print('Slice plots skipped as requested.')
    
    # Clean up temporary model file
    try:
        temp_keras_path = temp_model_path + '.keras'
        if os.path.exists(temp_keras_path):
            os.remove(temp_keras_path)
            print(f'Cleaned up temporary model file: {temp_keras_path}')
    except Exception as e:
        print(f'Warning: Could not remove temporary model file: {e}')
    
    print('\n=== Prediction and Analysis Complete ===')
    print(f'Results saved in: {MODEL_FIG_PATH}')
    if SAVE_PREDICTIONS:
        print(f'Raw predictions saved in: {PRED_PATH}')

if __name__ == '__main__':
    main()
