#!/usr/bin/env python3
'''
7/28/25: Implementing an idea about curricular training. Instead of loading a model designed 
for a specific interparticle separation (hereto referrred to as lambda and in units of Mpc/h),
we will instead begin the training on the lowest interparticle separation and
progressively increase the interparticle separation.

You can either score the models on the highest lambda or on the current lambda. 
'''
print('>>> Running curricular.py')
print('DEBUG: Script started successfully')

import os
import sys
print('DEBUG: Basic imports successful')

# Set environment variables for better memory management and stability
print('DEBUG: Setting TensorFlow environment variables...')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO and WARNING (but keep ERROR visible)
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # Better memory management
os.environ["TF_DISABLE_CUDNN_AUTOTUNE"] = "1"  # Disable autotuning for stability
os.environ["TF_DISABLE_SEGMENT_REDUCTION"] = "1"  # Disable problematic optimizations
os.environ["TF_ENABLE_EXPERIMENTAL_TENSOR_FLOAT_32_EXECUTION"] = "0"  # Disable TF32 for stability

# Additional CUDNN stability settings for A100
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"  # Disable CUDNN autotuning (consistent with TF_DISABLE_CUDNN_AUTOTUNE)
os.environ["CUDNN_LOGDEST_DBG"] = "/dev/null"  # Suppress CUDNN debug logs

# Additional stability variables
os.environ["TF_DETERMINISTIC_OPS"] = "0"  # Disable for performance (conflicts with frozen layers)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Force GPU memory growth
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"  # Better thread management
os.environ["CUDA_CACHE_DISABLE"] = "0"  # Enable CUDA kernel caching for performance
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"  # Disable XLA auto-jit (set explicitly later)

import argparse
import numpy as np
import tensorflow as tf
print('DEBUG: TensorFlow imported successfully')

# Suppress TensorFlow warnings and errors
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
tf.get_logger().setLevel('ERROR')
import warnings
warnings.filterwarnings('ignore')

import NETS_LITE as nets
import volumes
# Import validation functions
from NETS_LITE import MultiScaleValidationCallback, HybridValidationCallback
import plotter
import datetime
import gc
print('DEBUG: All imports completed successfully')
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
print('TensorFlow version:', tf.__version__)
print('CUDA?', tf.test.is_built_with_cuda())
# get the GPU devices
gpus = tf.config.list_physical_devices('GPU')
print('GPUs available:', gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Log GPU info without setting limits
if gpus:
    try:
        gpu_info = tf.config.experimental.get_device_details(gpus[0])
        device_name = gpu_info.get('device_name', 'Unknown')
        print(f'GPU device name: {device_name}')
    except Exception as e:
        print(f'Could not get GPU info: {e}')
nets.K.set_image_data_format('channels_last')
# NOTE turning off XLA JIT compilation for now, as it can cause issues with some models
tf.config.optimizer.set_jit(False)  # if you're using XLA
# Disable deterministic ops due to issues with frozen batch norm layers in TF 2.10.0
# tf.config.experimental.enable_op_determinism()  # Commented out - causes issues with frozen layers

# Additional workaround for libdevice issues on Picotte V100s
# Note: TF_CPP_MIN_LOG_LEVEL already set above
# Disable XLA clustering to avoid libdevice dependency while keeping GPUs enabled
tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})
from tensorflow.keras import mixed_precision

# Disable mixed precision on Picotte V100s to avoid NaN issues with custom loss functions
if os.getcwd().startswith('/ifs/groups/vogeleyGrp'):
    print('Detected Picotte environment - disabling mixed precision for V100 numerical stability')
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_global_policy(policy)
else:
    print('Not on Picotte - enabling mixed precision')
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
print(f'Mixed precision policy: {mixed_precision.global_policy().name}')
#===============================================================
# Set random seed
#===============================================================
seed = 12; print('Setting random seed:',seed)
np.random.seed(seed)
tf.random.set_seed(seed)
#================================================================
# Parse command line arguments
#================================================================
parser = argparse.ArgumentParser(description='Curricular training script for deep void detection.')
required = parser.add_argument_group('required arguments')
required.add_argument('ROOT_DIR', type=str, help='Root directory for the project.')
required.add_argument('DEPTH', type=int, default=3,
                      help='Depth of the model. Default is 3.')
required.add_argument('FILTERS', type=int, default=32,
                      help='Number of filters in the model. Default is 32.')
required.add_argument('LOSS', type=str, choices=['CCE', 'DISCCE', 'FOCAL_CCE', 'SCCE', 'SCCE_Void_Penalty', 'SCCE_Class_Penalty', 'SCCE_Balanced_Class_Penalty', 'SCCE_Class_Penalty_Fixed', 'SCCE_Proportion_Aware'],
                      help='Loss function to use for training.')
optional = parser.add_argument_group('optional arguments')
optional.add_argument('--UNIFORM_FLAG', action='store_true',
                      help='Use uniform mass subhalos for training data.')
optional.add_argument('--BATCH_SIZE', type=int, default=8, 
                      help='Batch size for training.')
optional.add_argument('--LEARNING_RATE', type=float, default=1e-4,
                      help='Learning rate for the optimizer.')
optional.add_argument('--LEARNING_RATE_PATIENCE', type=int, default=10,
                      help='Patience for learning rate reduction.')
optional.add_argument('--EARLY_STOP_PATIENCE', type=int, default=10,
                      help='Patience for early stopping.')
optional.add_argument('--EXTRA_INPUTS', type=str, default=None,
                      choices=['g-r', 'r_flux_density'],
                      help='Name of additional inputs for the model such as color or fluxes.')
optional.add_argument('--ADD_RSD', action='store_true',
                      help='Add RSD (Redshift Space Distortion) to the inputs.')
optional.add_argument('--L_VAL', type=str, default='10',
                      help='Interparticle separation for validation dataset. Default is 10.')
optional.add_argument('--VALIDATION_STRATEGY', type=str, default='target',
                      choices=['target', 'stage', 'hybrid', 'gradual'],
                      help='Validation strategy: target (validate on final goal), stage (validate on current training stage), hybrid (both), or gradual (progressive validation complexity). Default is target.')
optional.add_argument('--TARGET_LAMBDA', type=str, default=None,
                      help='Target lambda for final model performance (defaults to L_VAL if not specified).')
optional.add_argument('--USE_ATTENTION', action='store_true',
                      help='Use attention U-Net architecture instead of standard U-Net.')
optional.add_argument('--LAMBDA_CONDITIONING', action='store_true',
                      help='Use lambda conditioning in the model.')
optional.add_argument('--N_EPOCHS_PER_INTER_SEP', type=int, default=50,
                      help='Number of epochs to train for each interparticle separation. Default is 50.')
optional.add_argument('--PREPROCESSING', type=str, default='standard',
                      choices=['standard', 'robust', 'log_transform', 'clip_extreme'],
                      help='Preprocessing method for density data. Default is standard.')
optional.add_argument('--WARMUP_EPOCHS', type=int, default=0,
                      help='Number of warmup epochs with gradual learning rate increase. Default is 0 (no warmup).')
optional.add_argument('--FOCAL_ALPHA', nargs=4, type=float, default=[0.6, 0.3, 0.09, 0.02],
                      help='Alpha values for focal loss [void, wall, filament, halo]. Default: 0.6 0.3 0.09 0.02')
optional.add_argument('--FOCAL_GAMMA', type=float, default=1.5,
                      help='Gamma value for focal loss. Default: 1.5')
optional.add_argument('--USE_OVERLAPPING_SUBCUBES', action='store_true',
                      help='Enable overlapping subcubes with rotations for better data augmentation but higher memory usage. Default is to use non-overlapping subcubes for memory efficiency.')
optional.add_argument('--RSD_PRESERVING_ROTATIONS', action='store_true',
                      help='Use RSD-preserving rotations (only around z-axis and xy-flips) instead of full 3D rotations. Recommended when ADD_RSD is used to preserve line-of-sight anisotropy.')
optional.add_argument('--EXTRA_AUGMENTATION', action='store_true',
                      help='Enable extra data augmentation (doubles the number of rotated samples). Default is lighter augmentation for better memory usage.')
args = parser.parse_args()
ROOT_DIR = args.ROOT_DIR
DEPTH = args.DEPTH
FILTERS = args.FILTERS
LOSS = args.LOSS
UNIFORM_FLAG = args.UNIFORM_FLAG
BATCH_SIZE = args.BATCH_SIZE
LEARNING_RATE = args.LEARNING_RATE
LEARNING_RATE_PATIENCE = args.LEARNING_RATE_PATIENCE
EARLY_STOP_PATIENCE = args.EARLY_STOP_PATIENCE
EXTRA_INPUTS = args.EXTRA_INPUTS
ADD_RSD = args.ADD_RSD
L_VAL = args.L_VAL
USE_ATTENTION = args.USE_ATTENTION
LAMBDA_CONDITIONING = args.LAMBDA_CONDITIONING
N_EPOCHS_PER_INTER_SEP = args.N_EPOCHS_PER_INTER_SEP
PREPROCESSING = args.PREPROCESSING
WARMUP_EPOCHS = args.WARMUP_EPOCHS
FOCAL_ALPHA = args.FOCAL_ALPHA
FOCAL_GAMMA = args.FOCAL_GAMMA
USE_OVERLAPPING_SUBCUBES = args.USE_OVERLAPPING_SUBCUBES  # Default is False unless --USE_OVERLAPPING_SUBCUBES is specified
RSD_PRESERVING_ROTATIONS = args.RSD_PRESERVING_ROTATIONS  # Default is False unless --RSD_PRESERVING_ROTATIONS is specified
EXTRA_AUGMENTATION = args.EXTRA_AUGMENTATION  # Default is False unless --EXTRA_AUGMENTATION is specified

# Make RSD-preserving rotations the default when ADD_RSD is enabled (unless explicitly overridden)
if ADD_RSD and not args.RSD_PRESERVING_ROTATIONS and 'RSD_PRESERVING_ROTATIONS' not in sys.argv:
    RSD_PRESERVING_ROTATIONS = True
    print("Auto-enabling RSD-preserving rotations since --ADD_RSD is used")

# Set up validation strategy parameters
validation_strategy = args.VALIDATION_STRATEGY
target_lambda = args.TARGET_LAMBDA or args.L_VAL  # Default to L_VAL if not specified

print(f'Parsed arguments: ROOT_DIR={ROOT_DIR}, DEPTH={DEPTH}, FILTERS={FILTERS}, LOSS={LOSS}, UNIFORM_FLAG={UNIFORM_FLAG}, BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE}, LEARNING_RATE_PATIENCE={LEARNING_RATE_PATIENCE}, L_VAL={L_VAL}, USE_ATTENTION={USE_ATTENTION}, EXTRA_INPUTS={EXTRA_INPUTS}, ADD_RSD={ADD_RSD}, LAMBDA_CONDITIONING={LAMBDA_CONDITIONING}, N_EPOCHS_PER_INTER_SEP={N_EPOCHS_PER_INTER_SEP}, VALIDATION_STRATEGY={validation_strategy}, TARGET_LAMBDA={target_lambda}, PREPROCESSING={PREPROCESSING}, WARMUP_EPOCHS={WARMUP_EPOCHS}, FOCAL_ALPHA={FOCAL_ALPHA}, FOCAL_GAMMA={FOCAL_GAMMA}, USE_OVERLAPPING_SUBCUBES={USE_OVERLAPPING_SUBCUBES}, RSD_PRESERVING_ROTATIONS={RSD_PRESERVING_ROTATIONS}, EXTRA_AUGMENTATION={EXTRA_AUGMENTATION}')

# Validate RSD-related arguments
if RSD_PRESERVING_ROTATIONS and not ADD_RSD:
    print("WARNING: --RSD_PRESERVING_ROTATIONS is enabled but --ADD_RSD is not.")
    print("   RSD-preserving rotations are mainly beneficial when training with Redshift Space Distortions.")
    print("   Consider adding --ADD_RSD if your data contains line-of-sight distortions.")
    
if RSD_PRESERVING_ROTATIONS:
    print("Using RSD-preserving rotations: only z-axis rotations (90°, 180°, 270°) + xy-flips")
    print("   This preserves line-of-sight anisotropy but doubles the number of augmented samples (8x vs 4x)")
else:
    print("Using standard 3D rotations: rotations around all three axes (may disrupt RSD anisotropy)")

# Print subcube method information
if USE_OVERLAPPING_SUBCUBES:
    print("Using overlapping subcubes: better data augmentation but higher memory usage")
else:
    print("Using non-overlapping subcubes: memory efficient but less data augmentation")
    
# Validate focal loss parameters
if LOSS != 'FOCAL_CCE' and (args.FOCAL_ALPHA != [0.6, 0.3, 0.09, 0.02] or args.FOCAL_GAMMA != 1.5):
    print(f"WARNING: Focal loss parameters specified (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA}) but loss function is '{LOSS}', not 'FOCAL_CCE'. These parameters will be ignored.")

# Override mixed precision for attention models to avoid CUDNN precision issues
if USE_ATTENTION and not os.getcwd().startswith('/ifs/groups/vogeleyGrp'):
    print('WARNING: Attention model detected - disabling mixed precision to avoid CUDNN algorithm mismatch errors on A100')
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_global_policy(policy)
    print(f'Mixed precision policy overridden to: {mixed_precision.global_policy().name}')

# use mixed precision if on Picotte
if ROOT_DIR.startswith('/ifs/groups/vogeleyGrp/'):
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
#================================================================
# Set up custom objects for model serialization
#================================================================
CUSTOM_OBJECTS = {
    'MCC_keras': nets.MCC_keras,
    'F1_micro_keras': nets.F1_micro_keras,
    'void_F1_keras': nets.void_F1_keras,
    'SCCE_Dice_loss': nets.SCCE_Dice_loss,
    'categorical_focal_loss': nets.categorical_focal_loss,
    'SCCE_void_penalty': nets.SCCE_void_penalty,
    'SCCE_Class_Penalty': nets.SCCE_Class_Penalty,
    'VoidFractionMonitor': nets.VoidFractionMonitor,
    'RobustModelCheckpoint': nets.RobustModelCheckpoint,
    'Cast': tf.cast
}
#================================================================
# Set paths (according to Picotte data structure)
#================================================================
DATA_PATH = ROOT_DIR + 'data/TNG/'
FIG_PATH = ROOT_DIR + 'figs/TNG/'
MODEL_PATH = ROOT_DIR + 'models/'
PRED_PATH = ROOT_DIR + 'preds/'
if not os.path.exists(FIG_PATH):
    os.makedirs(FIG_PATH)
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if not os.path.exists(PRED_PATH):
    os.makedirs(PRED_PATH)
#================================================================
# Set filenames and parameters
#================================================================
BoxSize = 205.0 # Mpc/h
class_labels = ['void','wall','fila','halo']
N_CLASSES = 4
GRID = 512; SUBGRID = 128; OFF = 64
th = 0.65; # eigenvalue threshold
sig = 2.4 # smoothing scale
FILE_MASK = DATA_PATH + f'TNG300-3-Dark-mask-Nm={GRID}-th={th}-sig={sig}.fvol'
data_info = {
    '0.33' : 'DM_DEN_snap99_Nm=512.fvol',
    '3' : 'subs1_mass_Nm512_L3_d_None_smooth.fvol',
    '5' : 'subs1_mass_Nm512_L5_d_None_smooth.fvol',
    '7' : 'subs1_mass_Nm512_L7_d_None_smooth.fvol',
    '10' : 'subs1_mass_Nm512_L10_d_None_smooth.fvol',
}
inter_seps = list(data_info.keys())
if ADD_RSD:
    data_info['0.33'] = 'DM_DEN_snap99_perturbed_Nm=512.fvol'
    data_info['3'] = 'subs1_mass_Nm512_L3_RSD.fvol'
    data_info['5'] = 'subs1_mass_Nm512_L5_RSD.fvol'
    data_info['7'] = 'subs1_mass_Nm512_L7_RSD.fvol'
    data_info['10'] = 'subs1_mass_Nm512_L10_RSD.fvol'
if UNIFORM_FLAG:
    for key in data_info:
        if key != '0.33': # DM particles are already uniform
            data_info[key] = data_info[key].replace('.fvol', '_uniform.fvol')
# Add extra input files mapping
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
# Define loading function
#================================================================
def load_data(inter_sep, extra_inputs=None, verbose=True, preprocessing='standard'):
    '''
    Load the data for a given interparticle separation.
    Args:
        inter_sep (str): Interparticle separation (lambda) as a string.
        extra_inputs (str, optional): Additional inputs file name. Defaults to None.
        verbose (bool): Whether to print detailed loading statistics. Defaults to True.
        preprocessing (str): Preprocessing method for density data. Defaults to 'standard'.
    Returns:
        features (np.ndarray): Features array.
        labels (np.ndarray): Labels array.
        extra_input (np.ndarray, optional): Additional inputs array if provided.
    '''
    if inter_sep not in inter_seps:
        raise ValueError(f'Invalid interparticle separation: {inter_sep}. Must be one of {inter_seps}')
    
    data_file = DATA_PATH + data_info[inter_sep]
    if verbose:
        print(f'Loading data for L={inter_sep} Mpc/h from {data_file}')
    
    # Monitor GPU memory before loading
    if tf.config.list_physical_devices('GPU') and verbose:
        try:
            gpu_details = tf.config.experimental.get_memory_info('GPU:0')
            print(f'GPU memory before loading: {gpu_details["current"] / 1024**3:.2f} GB used')
        except:
            pass  # Memory info may not be available on all systems
    
    # Choose consistent subcube extraction method based on USE_OVERLAPPING_SUBCUBES flag
    if USE_OVERLAPPING_SUBCUBES:
        # Use overlapping subcubes with rotations for maximum data augmentation
        if RSD_PRESERVING_ROTATIONS:
            if EXTRA_AUGMENTATION:
                # Heavy RSD-preserving: 8x samples per subcube (4 z-rotations + 4 with xy-flips)
                features, labels = nets.load_dataset_all_overlap_rsd_preserving(
                    FILE_DEN=data_file,
                    FILE_MSK=FILE_MASK,
                    SUBGRID=SUBGRID,
                    OFF=OFF,
                    preprocessing=preprocessing
                )
                if verbose:
                    print(f'Using overlapping subcubes with heavy RSD-preserving rotations (8x per subcube)')
            else:
                # Light RSD-preserving: 4x samples per subcube (4 z-rotations only)
                features, labels = nets.load_dataset_all_overlap_rsd_preserving_light(
                    FILE_DEN=data_file,
                    FILE_MSK=FILE_MASK,
                    SUBGRID=SUBGRID,
                    OFF=OFF,
                    preprocessing=preprocessing
                )
                if verbose:
                    print(f'Using overlapping subcubes with light RSD-preserving rotations (4x per subcube)')
        else:
            if EXTRA_AUGMENTATION:
                # Heavy standard: 4x samples per subcube (full 3D rotations)
                features, labels = nets.load_dataset_all_overlap(
                    FILE_DEN=data_file,
                    FILE_MSK=FILE_MASK,
                    SUBGRID=SUBGRID,
                    OFF=OFF,
                    preprocessing=preprocessing
                )
                if verbose:
                    print(f'Using overlapping subcubes with heavy 3D rotations (4x per subcube)')
            else:
                # Light standard: 2x samples per subcube (original + one rotation)
                features, labels = nets.load_dataset_all_overlap_light(
                    FILE_DEN=data_file,
                    FILE_MSK=FILE_MASK,
                    SUBGRID=SUBGRID,
                    OFF=OFF,
                    preprocessing=preprocessing
                )
                if verbose:
                    print(f'Using overlapping subcubes with light augmentation (2x per subcube)')
    else:
        # Use non-overlapping subcubes with rotations for memory efficiency
        if RSD_PRESERVING_ROTATIONS:
            if EXTRA_AUGMENTATION:
                # Heavy RSD-preserving: 8x samples per subcube (4 z-rotations + 4 with xy-flips)
                features, labels = nets.load_dataset_all_rsd_preserving(
                    FILE_DEN=data_file,
                    FILE_MASK=FILE_MASK,
                    SUBGRID=SUBGRID,
                    verbose=verbose,
                    preprocessing=preprocessing
                )
                if verbose:
                    print(f'Using non-overlapping subcubes with heavy RSD-preserving rotations (8x per subcube)')
            else:
                # Light RSD-preserving: 4x samples per subcube (4 z-rotations only)
                features, labels = nets.load_dataset_all_rsd_preserving_light(
                    FILE_DEN=data_file,
                    FILE_MASK=FILE_MASK,
                    SUBGRID=SUBGRID,
                    verbose=verbose,
                    preprocessing=preprocessing
                )
                if verbose:
                    print(f'Using non-overlapping subcubes with light RSD-preserving rotations (4x per subcube)')
        else:
            if EXTRA_AUGMENTATION:
                # Heavy standard: 4x samples per subcube (full 3D rotations)
                features, labels = nets.load_dataset_all(
                    FILE_DEN=data_file,
                    FILE_MASK=FILE_MASK,
                    SUBGRID=SUBGRID,
                    verbose=verbose,
                    preprocessing=preprocessing
                )
                if verbose:
                    print(f'Using non-overlapping subcubes with heavy 3D rotations (4x per subcube)')
            else:
                # Light standard: 2x samples per subcube (original + one rotation)
                features, labels = nets.load_dataset_all_light(
                    FILE_DEN=data_file,
                    FILE_MASK=FILE_MASK,
                    SUBGRID=SUBGRID,
                    verbose=verbose,
                    preprocessing=preprocessing
                )
                if verbose:
                    print(f'Using non-overlapping subcubes with light augmentation (2x per subcube)')
    
    if extra_inputs is not None:
        extra_input_file = DATA_PATH + EXTRA_INPUTS_INFO[inter_sep]
        if verbose:
            print(f'Loading extra inputs from {extra_input_file}')
        
        # Use the same subcube method for consistency
        if USE_OVERLAPPING_SUBCUBES:
            # Use overlapping subcubes (like load_dataset) but ensure same number of samples as features
            extra_input = nets.load_dataset(extra_input_file, SUBGRID, OFF, preprocessing)
            
            if RSD_PRESERVING_ROTATIONS:
                # We need to replicate the data to match the 8 RSD-preserving rotations
                # RSD-preserving produces nbins³ × 8 samples, but load_dataset produces nbins³ samples
                # So we need to replicate each sample 8 times to match the rotation augmentation
                n_samples = extra_input.shape[0]
                extra_input_rotated = np.zeros((n_samples * 8, SUBGRID, SUBGRID, SUBGRID, 1), dtype=extra_input.dtype)
                
                for i in range(n_samples):
                    # Original
                    extra_input_rotated[i*8, :, :, :, 0] = extra_input[i, :, :, :, 0]
                    # 3 z-axis rotations
                    extra_input_rotated[i*8+1, :, :, :, 0] = volumes.rotate_cube(extra_input[i, :, :, :, 0], 2)
                    temp_rot = volumes.rotate_cube(extra_input[i, :, :, :, 0], 2)
                    extra_input_rotated[i*8+2, :, :, :, 0] = volumes.rotate_cube(temp_rot, 2)
                    temp_rot = volumes.rotate_cube(temp_rot, 2)
                    extra_input_rotated[i*8+3, :, :, :, 0] = volumes.rotate_cube(temp_rot, 2)
                    
                    # Same 4 rotations with xy-flip
                    flipped = np.flip(extra_input[i, :, :, :, 0], axis=0)
                    extra_input_rotated[i*8+4, :, :, :, 0] = flipped
                    extra_input_rotated[i*8+5, :, :, :, 0] = volumes.rotate_cube(flipped, 2)
                    temp_rot = volumes.rotate_cube(flipped, 2)
                    extra_input_rotated[i*8+6, :, :, :, 0] = volumes.rotate_cube(temp_rot, 2)
                    temp_rot = volumes.rotate_cube(temp_rot, 2)
                    extra_input_rotated[i*8+7, :, :, :, 0] = volumes.rotate_cube(temp_rot, 2)
                
                extra_input = extra_input_rotated
                if verbose:
                    print(f'Applied 8 RSD-preserving rotations to extra inputs to match main features')
            else:
                # We need to replicate the data to match the 4 rotations in load_dataset_all_overlap
                # load_dataset_all_overlap produces nbins³ × 4 samples, but load_dataset produces nbins³ samples
                # So we need to replicate each sample 4 times to match the rotation augmentation
                n_samples = extra_input.shape[0]
                extra_input_rotated = np.zeros((n_samples * 4, SUBGRID, SUBGRID, SUBGRID, 1), dtype=extra_input.dtype)
                
                for i in range(n_samples):
                    # Original
                    extra_input_rotated[i*4, :, :, :, 0] = extra_input[i, :, :, :, 0]
                    # 3 rotations (matching the rotation pattern in load_dataset_all_overlap)
                    extra_input_rotated[i*4+1, :, :, :, 0] = volumes.rotate_cube(extra_input[i, :, :, :, 0], 2)
                    extra_input_rotated[i*4+2, :, :, :, 0] = volumes.rotate_cube(extra_input[i, :, :, :, 0], 1) 
                    extra_input_rotated[i*4+3, :, :, :, 0] = volumes.rotate_cube(extra_input[i, :, :, :, 0], 0)
                
                extra_input = extra_input_rotated
                if verbose:
                    print(f'Applied 4 rotations to extra inputs to match main features')
        else:
            # Use non-overlapping subcubes - need to create a custom loader for extra inputs
            # that matches the non-overlapping pattern of load_dataset_all but for single files
            den = volumes.read_fvolume(extra_input_file)
            
            # Apply preprocessing
            if preprocessing == 'log_transform':
                den = np.log10(den + 1e-6)
                den = (den - np.mean(den)) / (np.std(den) + 1e-8)
            elif preprocessing == 'standard' or preprocessing is None:
                den = nets.minmax(den)
            # Add other preprocessing methods as needed
            
            # Extract non-overlapping subcubes with rotations (matching load_dataset_all pattern)
            n_bins = den.shape[0] // SUBGRID
            if RSD_PRESERVING_ROTATIONS:
                if EXTRA_AUGMENTATION:
                    # Heavy RSD-preserving: 8 rotations (4 z-axis + 4 z-axis with xy-flip)
                    extra_input = np.zeros(((n_bins**3)*8, SUBGRID, SUBGRID, SUBGRID, 1), dtype=np.float32)
                    
                    cont = 0
                    for i in range(n_bins):
                        for j in range(n_bins):
                            for k in range(n_bins):
                                sub_den = den[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
                                
                                # Original + 3 z-axis rotations
                                extra_input[cont,:,:,:,0] = sub_den
                                extra_input[cont+1,:,:,:,0] = volumes.rotate_cube(sub_den.copy(), 2)
                                temp_rot = volumes.rotate_cube(sub_den.copy(), 2)
                                extra_input[cont+2,:,:,:,0] = volumes.rotate_cube(temp_rot, 2)
                                temp_rot = volumes.rotate_cube(temp_rot, 2)
                                extra_input[cont+3,:,:,:,0] = volumes.rotate_cube(temp_rot, 2)
                                
                                # Same 4 rotations with xy-flip
                                sub_den_flip = np.flip(sub_den, axis=0)
                                extra_input[cont+4,:,:,:,0] = sub_den_flip
                                extra_input[cont+5,:,:,:,0] = volumes.rotate_cube(sub_den_flip.copy(), 2)
                                temp_rot = volumes.rotate_cube(sub_den_flip.copy(), 2)
                                extra_input[cont+6,:,:,:,0] = volumes.rotate_cube(temp_rot, 2)
                                temp_rot = volumes.rotate_cube(temp_rot, 2)
                                extra_input[cont+7,:,:,:,0] = volumes.rotate_cube(temp_rot, 2)
                                
                                cont += 8
                    
                    if verbose:
                        print(f'Applied 8 heavy RSD-preserving rotations to extra inputs for non-overlapping subcubes')
                else:
                    # Light RSD-preserving: 4 rotations (4 z-axis only) - MATCHES load_dataset_all_rsd_preserving_light
                    extra_input = np.zeros(((n_bins**3)*4, SUBGRID, SUBGRID, SUBGRID, 1), dtype=np.float32)
                    
                    cont = 0
                    for i in range(n_bins):
                        for j in range(n_bins):
                            for k in range(n_bins):
                                sub_den = den[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
                                
                                # Only 4 z-axis rotations (matching light RSD-preserving)
                                extra_input[cont,:,:,:,0] = sub_den
                                extra_input[cont+1,:,:,:,0] = volumes.rotate_cube(sub_den.copy(), 2)
                                temp_rot = volumes.rotate_cube(sub_den.copy(), 2)
                                extra_input[cont+2,:,:,:,0] = volumes.rotate_cube(temp_rot, 2)
                                temp_rot = volumes.rotate_cube(temp_rot, 2)
                                extra_input[cont+3,:,:,:,0] = volumes.rotate_cube(temp_rot, 2)
                                cont += 4
                    
                    if verbose:
                        print(f'Applied 4 light RSD-preserving rotations to extra inputs for non-overlapping subcubes')
            else:
                if EXTRA_AUGMENTATION:
                    # Heavy standard: 4 rotations (around all axes)
                    extra_input = np.zeros(((n_bins**3)*4, SUBGRID, SUBGRID, SUBGRID, 1), dtype=np.float32)
                    
                    cont = 0
                    for i in range(n_bins):
                        for j in range(n_bins):
                            for k in range(n_bins):
                                sub_den = den[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
                                # Original + 3 rotations (matching load_dataset_all pattern exactly)
                                extra_input[cont,:,:,:,0] = sub_den
                                extra_input[cont+1,:,:,:,0] = volumes.rotate_cube(sub_den, 2)
                                extra_input[cont+2,:,:,:,0] = volumes.rotate_cube(sub_den, 1)
                                extra_input[cont+3,:,:,:,0] = volumes.rotate_cube(sub_den, 0)
                                cont += 4
                    
                    if verbose:
                        print(f'Applied 4 heavy traditional rotations to extra inputs for non-overlapping subcubes')
                else:
                    # Light standard: 2 rotations - MATCHES load_dataset_all_light
                    extra_input = np.zeros(((n_bins**3)*2, SUBGRID, SUBGRID, SUBGRID, 1), dtype=np.float32)
                    
                    cont = 0
                    for i in range(n_bins):
                        for j in range(n_bins):
                            for k in range(n_bins):
                                sub_den = den[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
                                # Only 2 rotations (matching light augmentation)
                                extra_input[cont,:,:,:,0] = sub_den
                                extra_input[cont+1,:,:,:,0] = volumes.rotate_cube(sub_den, 2)
                                cont += 2
                    
                    if verbose:
                        print(f'Applied 2 light rotations to extra inputs for non-overlapping subcubes')
            
            if verbose:
                print(f'Extracted non-overlapping subcubes with 4 rotations for extra inputs')
        
        if verbose:
            print(f'Extra input shape: {extra_input.shape}')
        # Concatenate extra inputs to features
        features = np.concatenate([features, extra_input], axis=-1)
        if verbose:
            print(f'Features shape after concatenation: {features.shape}')
    
    if verbose:
        subcube_method = "overlapping with rotations" if USE_OVERLAPPING_SUBCUBES else "non-overlapping with rotations"
        print(f'Data loaded successfully using {subcube_method} subcubes.')
        print(f'Final features: {features.shape}, Labels: {labels.shape}')
    
    return features, labels

def make_dataset(delta, tij_labels, batch_size=BATCH_SIZE, shuffle=True, one_hot=False, lambda_value=None):
    '''
    Create a TensorFlow dataset from the features and labels with memory-efficient handling.
    '''
    print(f'Creating dataset with {len(delta)} samples...')
    print(f'Features shape: {delta.shape}, Labels shape: {tij_labels.shape}')
    if one_hot:
        tij_labels = tf.keras.utils.to_categorical(tij_labels, num_classes=N_CLASSES)
    
    # Check for NaN values in delta and tij_labels
    if np.any(np.isnan(delta)):
        print('WARNING: NaN values found in features!')
        delta = np.nan_to_num(delta, nan=0.0)
    
    if np.any(np.isnan(tij_labels)):
        print('WARNING: NaN values found in labels!')
        tij_labels = np.nan_to_num(tij_labels, nan=0.0)
    
    # Memory-efficient tensor conversion
    try:
        print(f'Creating dataset with batch size {batch_size} and shuffle={shuffle}...')
        
        if lambda_value is not None and LAMBDA_CONDITIONING:
            # Create lambda conditioning tensor with the same batch size as features
            lambda_tensor = np.full((delta.shape[0], 1), lambda_value, dtype=np.float32)
            
            # For lambda conditioning, model expects [inputs, lambda_input] and outputs [segmentation, lambda_pred]
            # We need to provide labels for both: [segmentation_labels, lambda_labels]
            lambda_labels = np.full((tij_labels.shape[0], 1), lambda_value, dtype=np.float32)
            
            dataset = tf.data.Dataset.from_tensor_slices(((delta, lambda_tensor), (tij_labels, lambda_labels)))
            
        else:
            dataset = tf.data.Dataset.from_tensor_slices((delta, tij_labels))
        
        # Apply transformations
        if shuffle:
            buffer_size = min(10000, len(delta))  # Cap shuffle buffer for memory efficiency
            dataset = dataset.shuffle(buffer_size, seed=42, reshuffle_each_iteration=True)
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
        
    except tf.errors.ResourceExhaustedError as e:
        print(f'GPU memory exhausted, trying with smaller batch size: {e}')
        # Reduce batch size and try again
        reduced_batch_size = max(1, batch_size // 2)
        print(f'Retrying with batch size {reduced_batch_size}...')
        return make_dataset(delta, tij_labels, reduced_batch_size, shuffle, one_hot, lambda_value)
        
    except Exception as e:
        print(f'Error creating dataset: {e}')
        print('Falling back to CPU-based tensor conversion...')
        
        # Fallback: explicit CPU tensor creation
        with tf.device('/CPU:0'):
            delta_tensor = tf.convert_to_tensor(delta, dtype=tf.float32)
            labels_tensor = tf.convert_to_tensor(tij_labels, dtype=tf.float32)
        
        if lambda_value is not None and LAMBDA_CONDITIONING:
            with tf.device('/CPU:0'):
                lambda_tensor = tf.fill([tf.shape(delta_tensor)[0], 1], tf.cast(lambda_value, tf.float32))
                lambda_labels = tf.fill([tf.shape(labels_tensor)[0], 1], tf.cast(lambda_value, tf.float32))
            
            # Create dataset with multiple inputs and outputs for lambda conditioning
            dataset = tf.data.Dataset.from_tensor_slices(((delta_tensor, lambda_tensor), (labels_tensor, lambda_labels)))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((delta_tensor, labels_tensor))
        
        if shuffle:
            buffer_size = min(10000, len(delta))
            dataset = dataset.shuffle(buffer_size, seed=42, reshuffle_each_iteration=True)
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
#================================================================
# Create model parameters
#================================================================
SIM = 'TNG'
MODEL_NAME = f'{SIM}_curricular_{LOSS}_D{DEPTH}_F{FILTERS}'
if UNIFORM_FLAG:
    MODEL_NAME += '_uniform'
if ADD_RSD:
    MODEL_NAME += '_RSD'
if RSD_PRESERVING_ROTATIONS:
    MODEL_NAME += '_RSDrot'
if USE_ATTENTION:
    MODEL_NAME += '_attention'
if LAMBDA_CONDITIONING:
    MODEL_NAME += '_lambda'
if EXTRA_INPUTS:
    MODEL_NAME += f'_{EXTRA_INPUTS}'
print(f'Model name stem: {MODEL_NAME}')
DATE = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
MODEL_NAME += f'_{DATE}'
print(f'Model name: {MODEL_NAME}')
last_activation = 'softmax' # NOTE implement changing later 
input_shape = (None, None, None, 1)
if EXTRA_INPUTS:
    input_shape = (None, None, None, 1 + 1) # Adjust for extra inputs
print(f'Input shape: {input_shape}')
#================================================================
# Create validation datasets based on strategy
#================================================================
if 'SCCE' in LOSS or 'DISCCE' in LOSS:
    ONE_HOT = False
else:
    ONE_HOT = True
print(f'ONE_HOT encoding: {ONE_HOT}')

# Create validation datasets based on strategy
print(f'Creating validation datasets with strategy: {validation_strategy}')
print('DEBUG: About to set up validation strategy (not loading data yet)')

# For universal compatibility, use lazy loading for validation datasets
def create_validation_dataset(lambda_val, cache_key=None):
    """Create validation dataset with memory management"""
    print(f'Loading validation data for L={lambda_val} Mpc/h...')
    x_val, y_val = load_data(lambda_val, extra_inputs=EXTRA_INPUTS, verbose=False, preprocessing=PREPROCESSING)
    val_dataset = make_dataset(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False, one_hot=ONE_HOT, 
                              lambda_value=float(lambda_val) if LAMBDA_CONDITIONING else None)
    # Clear memory immediately after dataset creation
    del x_val, y_val
    gc.collect()
    return val_dataset

# Define gradual validation mapping: stage lambda -> validation lambda
gradual_validation_map = {
    '0.33': '0.33',  # Base density validates on base density
    '3': '5',        # First subhalo stage validates on intermediate complexity
    '5': '5',        # Intermediate stage validates on itself
    '7': '10',       # Higher stages validate on final complexity
    '10': '10'       # Final stage validates on final complexity
}

print('DEBUG: Validation mapping defined, setting up strategy without loading data')

if validation_strategy == 'gradual':
    print('Gradual validation mapping:')
    for stage_lambda, val_lambda in gradual_validation_map.items():
        print(f'  Training L={stage_lambda} -> Validation L={val_lambda}')

# Delay validation dataset creation until training loop starts
val_dataset = None
print('DEBUG: Validation dataset creation deferred until training starts')
print('DEBUG: Skipping early validation dataset loading to avoid RAM spike')
#================================================================
# Set loss function and metrics
#================================================================
print('DEBUG: Setting up loss function and metrics')
metrics = ['accuracy']
metrics += [nets.MCC_keras(int_labels=not ONE_HOT),
            nets.F1_micro_keras(int_labels=not ONE_HOT),
            nets.void_F1_keras(int_labels=not ONE_HOT)]

# Define loss function based on LOSS argument
if LOSS == 'CCE':
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
elif LOSS == 'DISCCE':
    loss_fn = nets.SCCE_Dice_loss
elif LOSS == 'FOCAL_CCE':
    alpha = FOCAL_ALPHA
    gamma = FOCAL_GAMMA
    loss_fn = nets.categorical_focal_loss(alpha=alpha, gamma=gamma)
    print(f'Using Focal Loss with alpha={alpha} and gamma={gamma}')
elif LOSS == 'SCCE':
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
elif LOSS == 'SCCE_Void_Penalty':
    loss_fn = nets.SCCE_void_penalty
elif LOSS == 'SCCE_Class_Penalty':
    # Use the enhanced class penalty function with balanced parameters
    def scce_class_penalty_loss(y_true, y_pred):
        # Reduced void_penalty from 8.0 to 0.5 to prevent over-penalizing void predictions
        # Reduced minority_boost from 3.0 to 1.5 to prevent extreme bias toward wall
        return nets.SCCE_Class_Penalty(y_true, y_pred, void_penalty=0.5, minority_boost=1.5)
    
    loss_fn = scce_class_penalty_loss
    # Add the custom loss function to the custom objects dictionary
    CUSTOM_OBJECTS['scce_class_penalty_loss'] = scce_class_penalty_loss
elif LOSS == 'SCCE_Balanced_Class_Penalty':
    # Use the new balanced class penalty function
    def scce_balanced_class_penalty_loss(y_true, y_pred):
        return nets.SCCE_Balanced_Class_Penalty(y_true, y_pred, void_penalty=1.5, wall_penalty=1.5, minority_boost=2.0)
    
    loss_fn = scce_balanced_class_penalty_loss
    # Add the custom loss function to the custom objects dictionary
    CUSTOM_OBJECTS['scce_balanced_class_penalty_loss'] = scce_balanced_class_penalty_loss
elif LOSS == 'SCCE_Class_Penalty_Fixed':
    # Use the improved fixed class penalty function
    def scce_class_penalty_fixed_loss(y_true, y_pred):
        return nets.SCCE_Class_Penalty_Fixed(y_true, y_pred, void_penalty=2.0, wall_penalty=1.0, minority_boost=2.0)
    
    loss_fn = scce_class_penalty_fixed_loss
    CUSTOM_OBJECTS['scce_class_penalty_fixed_loss'] = scce_class_penalty_fixed_loss
elif LOSS == 'SCCE_Proportion_Aware':
    # Use the proportion-aware loss function
    def scce_proportion_aware_loss(y_true, y_pred):
        return nets.SCCE_Proportion_Aware(y_true, y_pred, target_props=[0.65, 0.25, 0.08, 0.02], prop_weight=1.0)
    
    loss_fn = scce_proportion_aware_loss
    CUSTOM_OBJECTS['scce_proportion_aware_loss'] = scce_proportion_aware_loss
# Make tensorboard directory
log_dir = ROOT_DIR + 'logs/fit/' + MODEL_NAME + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '/'
print(f"DEBUG: Creating log directory: {log_dir}")
os.makedirs(log_dir, exist_ok=True)
print("DEBUG: Log directory created successfully")
#================================================================
# Create the model
#================================================================
print(f'Creating model with depth={DEPTH}, filters={FILTERS}, loss={LOSS}, uniform={UNIFORM_FLAG}, RSD={ADD_RSD}, attention={USE_ATTENTION}...')
print('DEBUG: About to create model - this might cause RAM spike')
#strategy = tf.distribute.MirroredStrategy()
strategy = tf.distribute.get_strategy()
print('Number of devices:', strategy.num_replicas_in_sync)
print('DEBUG: Strategy created, about to enter strategy scope')
with strategy.scope():
    print('DEBUG: Inside strategy scope, about to create model')
    if USE_ATTENTION:
        model = nets.attention_unet_3d(
            input_shape=input_shape,
            num_classes=N_CLASSES,
            initial_filters=FILTERS,
            depth=DEPTH,
            last_activation=last_activation,
            batch_normalization=True,
            model_name=MODEL_NAME,
            lambda_conditioning=LAMBDA_CONDITIONING
        )
    else:
        model = nets.unet_3d(
            input_shape=input_shape,
            num_classes=N_CLASSES,
            initial_filters=FILTERS,
            depth=DEPTH,
            last_activation=last_activation,
            batch_normalization=True,
            model_name=MODEL_NAME,
            lambda_conditioning=LAMBDA_CONDITIONING
        )
    print('DEBUG: Model architecture created successfully')
    # Create optimizer with gradient clipping to prevent NaN losses
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, 
        clipnorm=1.0  # Use gradient norm clipping only
    )
    print('DEBUG: Optimizer created, about to compile model')
    
    if LAMBDA_CONDITIONING:
        # For lambda conditioning, the model outputs [segmentation, lambda_pred]
        model.compile(
            optimizer=optimizer,
            loss=[loss_fn, 'mse'],  # Use list format for multiple outputs
            loss_weights=[1.0, 0.1],  # Main loss gets full weight, lambda loss gets 0.1
            metrics=[metrics, 'mse']  # Segmentation metrics and lambda MSE
        )
        print('DEBUG: Model compiled successfully (LAMBDA_CONDITIONING branch)')
    else:
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )
        print('DEBUG: Model compiled successfully (normal branch)')
print("DEBUG: About to call model.summary() - this might be where it hangs...")
# Temporarily disable model.summary() to test if this is causing the hang
# print(model.summary())
print("DEBUG: Skipped model.summary() call - testing if this was causing the hang")
print("DEBUG: Model summary completed successfully - about to define callbacks")
print("DEBUG: Python interpreter still responsive, continuing...")
#================================================================
# Learning Rate Warmup Callback
#================================================================
print("DEBUG: Defining WarmupLearningRateScheduler class...")
class WarmupLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs, target_lr, verbose=1):
        super(WarmupLearningRateScheduler, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.verbose = verbose
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Gradual warmup from target_lr/10 to target_lr
            warmup_lr = self.target_lr * (0.1 + 0.9 * (epoch + 1) / self.warmup_epochs)
            # Direct access to optimizer (no more LossScaleOptimizer)
            optimizer = self.model.optimizer
                
            # Use the most compatible method to set learning rate
            try:
                # Try the newer method first
                optimizer.learning_rate.assign(warmup_lr)
            except AttributeError:
                try:
                    # Fallback to older method
                    tf.keras.backend.set_value(optimizer.learning_rate, warmup_lr)
                except Exception as e:
                    print(f"Warning: Could not set learning rate during warmup: {e}")
                    
            if self.verbose:
                print(f'\nWarmup epoch {epoch + 1}/{self.warmup_epochs}: Learning rate set to {warmup_lr:.6f}')
        elif epoch == self.warmup_epochs:
            # Set to target learning rate after warmup
            optimizer = self.model.optimizer
                
            try:
                optimizer.learning_rate.assign(self.target_lr)
            except AttributeError:
                try:
                    tf.keras.backend.set_value(optimizer.learning_rate, self.target_lr)
                except Exception as e:
                    print(f"Warning: Could not set target learning rate: {e}")
            if self.verbose:
                print(f'\nWarmup complete. Learning rate set to target: {self.target_lr:.6f}')

print("DEBUG: WarmupLearningRateScheduler class defined successfully")
#================================================================
# Training loop
#================================================================
print('>>> Starting curricular training...')
print("DEBUG: Entering main training loop")
print("DEBUG: About to create ReduceLROnPlateau callback...")
print("DEBUG: Importing ReduceLROnPlateau from tensorflow.keras.callbacks...")
reduce_LR = ReduceLROnPlateau(
            patience=LEARNING_RATE_PATIENCE,
            factor=0.5,
            monitor='val_loss',
            mode='min',
            verbose=1,
            min_lr=1e-6
)
print("DEBUG: ReduceLROnPlateau callback created successfully")
# set freezing scheme:
print("DEBUG: Setting up density_to_freeze_map...")
density_to_freeze_map = {
    '0.33': 0,  # No freezing for the lowest interparticle separation
    '3': 0,     # Freeze first block for L=3 Mpc/h
    '5': 1,     # Freeze first two blocks for L=5 Mpc/h
    '7': 2,     # Freeze first three blocks for L=7 Mpc/h
    '10': 3     # Freeze first four blocks for L=10 Mpc/h
}
print("DEBUG: density_to_freeze_map created successfully")
# create combined history object to store metrics for all interparticle separations
print("DEBUG: Creating combined_history object...")
combined_history = {
    'loss': [],
    'val_loss': [],
    'accuracy': [],
    'val_accuracy': [],
    'f1_micro': [],
    'val_f1_micro': [],
    'mcc': [],
    'val_mcc': [],
    'void_f1': [],
    'val_void_f1': [],
    'void_fraction': [],
    'val_void_fraction': [],
    'lr': [],
    'epoch': []
}
print("DEBUG: combined_history created successfully")
print("DEBUG: Creating TensorBoard callback...")
tensor_board_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=False,
    update_freq='epoch',
)
print("DEBUG: TensorBoard callback created successfully")
#================================================================
# Training loop over interparticle separations
#================================================================
print('>>> Starting training loop over interparticle separations...')
print("DEBUG: About to check validation strategy for hybrid initialization...")

# Initialize validation callback for hybrid strategy
validation_callback = None
print(f"DEBUG: Current validation_strategy = '{validation_strategy}'")
if validation_strategy == 'hybrid':
    print("DEBUG: Entering hybrid validation setup...")
    # Pre-create target validation dataset for hybrid validation
    print(f'Creating target validation dataset for L={target_lambda} Mpc/h...')
    target_val_dataset = create_validation_dataset(target_lambda)
    
    # Pre-create stage validation datasets for all interparticle separations
    stage_datasets = {}
    print('Creating stage validation datasets for hybrid validation...')
    for sep in inter_seps:
        print(f'  Creating validation dataset for L={sep} Mpc/h...')
        stage_datasets[sep] = create_validation_dataset(float(sep))
        
    validation_callback = HybridValidationCallback(
        target_dataset=target_val_dataset,
        stage_datasets=stage_datasets,
        inter_seps=inter_seps,
        verbose=1
    )
    print(f'Initialized HybridValidationCallback with {len(stage_datasets)} stage datasets')
else:
    print(f"DEBUG: Skipping hybrid validation setup for strategy '{validation_strategy}'")

print("DEBUG: About to start main training loop...")
epoch_offset = 0
for i, inter_sep in enumerate(inter_seps):
    print(f'Starting training for interparticle separation L={inter_sep} Mpc/h (stage {i+1}/{len(inter_seps)})...')
    
    # Aggressive memory cleanup before each new stage (including first stage for safety)
    if i > 0:
        print(f'Performing aggressive memory cleanup before L={inter_sep} Mpc/h stage...')
        
        # Delete training dataset from previous iteration
        try:
            del train_dataset
            print('Previous training dataset deleted')
        except:
            pass
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # Clear TensorFlow session and backend state
        tf.keras.backend.clear_session()
        
        # Force GPU memory cleanup
        if tf.config.list_physical_devices('GPU'):
            try:
                # Reset GPU memory allocation
                gpus = tf.config.list_physical_devices('GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print('GPU memory growth reset')
            except Exception as e:
                print(f'GPU memory reset failed (non-critical): {e}')
        
        print('Memory cleanup completed')
        
        # Monitor memory after cleanup
        try:
            import psutil
            memory_usage = psutil.virtual_memory()
            print(f'System RAM after cleanup: {memory_usage.used / 1024**3:.2f} GB used / {memory_usage.total / 1024**3:.2f} GB total ({memory_usage.percent:.1f}%)')
        except:
            pass
    # Only show detailed stats for the first data load
    verbose_load = (i == 0)
    
    # Monitor memory before data loading
    try:
        import psutil
        memory_before = psutil.virtual_memory()
        print(f'System RAM before loading L={inter_sep}: {memory_before.used / 1024**3:.2f} GB used / {memory_before.total / 1024**3:.2f} GB total ({memory_before.percent:.1f}%)')
    except:
        pass
    
    if EXTRA_INPUTS is not None:
        train_features, train_labels = load_data(inter_sep, extra_inputs=EXTRA_INPUTS, verbose=verbose_load, preprocessing=PREPROCESSING)
    else:
        train_features, train_labels = load_data(inter_sep, verbose=verbose_load, preprocessing=PREPROCESSING)
    
    # Monitor memory after data loading
    try:
        import psutil
        memory_after = psutil.virtual_memory()
        memory_increase = (memory_after.used - memory_before.used) / 1024**3
        print(f'System RAM after loading L={inter_sep}: {memory_after.used / 1024**3:.2f} GB used / {memory_after.total / 1024**3:.2f} GB total ({memory_after.percent:.1f}%)')
        print(f'Memory increase from data loading: +{memory_increase:.2f} GB')
    except:
        pass
        
    print(f'Training data loaded for L={inter_sep}. Features shape: {train_features.shape}, Labels shape: {train_labels.shape}')
    # Create the training dataset
    train_dataset = make_dataset(train_features, train_labels, batch_size=BATCH_SIZE, shuffle=True, one_hot=ONE_HOT, lambda_value=float(inter_sep) if LAMBDA_CONDITIONING else None)
    
    # Clear training data from memory after dataset creation
    del train_features, train_labels
    
    # Force garbage collection multiple times to ensure cleanup
    for _ in range(3):
        gc.collect()
    
    # Monitor memory after dataset creation and cleanup
    try:
        import psutil
        memory_after_dataset = psutil.virtual_memory()
        print(f'System RAM after dataset creation & cleanup: {memory_after_dataset.used / 1024**3:.2f} GB used / {memory_after_dataset.total / 1024**3:.2f} GB total ({memory_after_dataset.percent:.1f}%)')
    except:
        pass
        
    print('Training data cleared from memory after dataset creation')
    
    # Monitor GPU memory after data loading
    try:
        gpu_details = tf.config.experimental.get_memory_info('GPU:0')
        print(f'GPU memory after loading L={inter_sep}: {gpu_details["current"] / 1024**3:.2f} GB used')
    except:
        pass
    
    # Update validation dataset for stage-based validation
    if validation_strategy == 'stage':
        print(f'Updating validation dataset for stage-based validation: L={inter_sep} Mpc/h')
        # Clear old validation dataset to free memory
        if 'val_dataset' in locals() and val_dataset is not None:
            del val_dataset
            gc.collect()
        val_dataset = create_validation_dataset(inter_sep)
    
    # Update validation dataset for target validation strategy (create once at the beginning)
    elif validation_strategy == 'target':
        if i == 0:  # Only create on the first iteration
            print(f'Creating target validation dataset: L={target_lambda} Mpc/h')
            val_dataset = create_validation_dataset(target_lambda)
        # Otherwise, keep using the same validation dataset
    
    # Update validation dataset for gradual validation strategy
    elif validation_strategy == 'gradual':
        current_val_lambda = gradual_validation_map[inter_sep]
        if i == 0 or current_val_lambda != gradual_validation_map[inter_seps[i-1]]:
            print(f'Updating validation dataset for gradual validation: training L={inter_sep} -> validation L={current_val_lambda} Mpc/h')
            # Clear old validation dataset to free memory
            if 'val_dataset' in locals() and val_dataset is not None:
                del val_dataset
                gc.collect()
            val_dataset = create_validation_dataset(current_val_lambda)
        else:
            print(f'Keeping current validation dataset: training L={inter_sep} -> validation L={current_val_lambda} Mpc/h')
    
    # Update stage for hybrid validation
    elif validation_strategy == 'hybrid' and validation_callback is not None:
        print(f'Updating hybrid validation stage: L={inter_sep} Mpc/h')
        validation_callback.update_stage(i, inter_sep)
    # freeze layers based on interparticle separation. freeze more layers for larger interparticle separations:
    nets.freeze_encoder_blocks(
        model,
        density_to_freeze_map[inter_sep],
    )
    if inter_sep != '0.33':
        # Clear optimizer state before recompiling to prevent memory accumulation
        print(f'Resetting optimizer state before recompiling for L={inter_sep}')
        
        # recompile the model after freezing layers
        if LAMBDA_CONDITIONING:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
                loss=[loss_fn, 'mse'],  # Use list format for multiple outputs
                loss_weights=[1.0, 0.1],  # Main loss gets full weight, lambda loss gets 0.1
                metrics=[metrics, 'mse'])  # Segmentation metrics and lambda MSE
        else:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
                loss=loss_fn,
                metrics=metrics)
        print(f'Model recompiled after freezing layers for interparticle separation L={inter_sep}')
    # Callbacks:
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOP_PATIENCE,
        mode='min',
        verbose=1,
        restore_best_weights=True
    )
    void_fraction_monitor = nets.VoidFractionMonitor(
        val_dataset=val_dataset,
        max_batches=16
    )
    checkpt_path = MODEL_PATH + MODEL_NAME + f'_L{inter_sep}.keras'
    weights_path = MODEL_PATH + MODEL_NAME + f'_L{inter_sep}_weights.h5'
    
    callbacks = [
        nets.RobustModelCheckpoint(
            model_path=checkpt_path,
            weights_path=weights_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        tensor_board_callback,
        reduce_LR,
        early_stop,
        void_fraction_monitor
    ]
    
    # Add learning rate warmup if specified
    if WARMUP_EPOCHS > 0 and i == 0:  # Only add warmup for the first interparticle separation
        warmup_callback = WarmupLearningRateScheduler(
            warmup_epochs=WARMUP_EPOCHS,
            target_lr=LEARNING_RATE,
            verbose=1
        )
        callbacks.insert(1, warmup_callback)  # Insert after checkpoint callback
        print(f'Added learning rate warmup for {WARMUP_EPOCHS} epochs')
    
    # Add multi-scale validation callback for hybrid validation strategy
    if validation_strategy == 'hybrid' and validation_callback is not None:
        callbacks.append(validation_callback)
    # add printing # of parameters:
    total_params = model.count_params()
    print(f'Total number of parameters in the model: {total_params}')
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
    print(f'Trainable parameters: {trainable_params}')
    non_trainable_params = total_params - trainable_params
    print(f'Non-trainable parameters: {non_trainable_params}')
    
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=N_EPOCHS_PER_INTER_SEP,  # Set a reasonable number of epochs
        callbacks=callbacks,
        verbose=2
    )

    # Always load the best model weights after training
    weights_path = MODEL_PATH + MODEL_NAME + f'_L{inter_sep}_weights.weights.h5'
    legacy_weights_path = MODEL_PATH + MODEL_NAME + f'_L{inter_sep}_weights.h5'  # Fallback for older format
    
    # Try weights first (more reliable for complex models with lambda conditioning)
    if os.path.exists(weights_path):
        print(f'Loading best model weights from {weights_path}')
        try:
            model.load_weights(weights_path)
            print('Best model weights loaded successfully from weights file.')
        except Exception as e:
            print(f'Failed to load weights: {e}')
            print('Continuing with current model state...')
    elif os.path.exists(legacy_weights_path):
        print(f'New weights file not found, trying legacy format: {legacy_weights_path}')
        try:
            model.load_weights(legacy_weights_path)
            print('Best model weights loaded successfully from legacy weights file.')
        except Exception as e:
            print(f'Failed to load legacy weights: {e}')
            print('Continuing with current model state...')
    elif os.path.exists(checkpt_path):
        print(f'Weights file not found, attempting to load full model from {checkpt_path}')
        try:
            best_model = tf.keras.models.load_model(checkpt_path, custom_objects=CUSTOM_OBJECTS, compile=False)
            model.set_weights(best_model.get_weights())
            print('Best model weights loaded from full model.')
            del best_model  # Free memory
        except Exception as e:
            print(f'Failed to load full model: {e}')
            print('Continuing with current model state...')
    else:
        print('No checkpoint found, continuing with current model state...')

    # Append the history to the combined history
    for key in combined_history.keys():
        if key in history.history:
            combined_history[key].extend(history.history[key])
    print(f'Combined history now has {len(combined_history["loss"])} total epochs')

    # update epochs
    actual_epochs = len(history.history['loss'])
    epoch_numbers = list((range(epoch_offset, epoch_offset + actual_epochs)))
    combined_history['epoch'].extend(epoch_numbers)
    epoch_offset += actual_epochs
    print(f'Combined history now has {len(combined_history["epoch"])} total epochs')
    print(f'Current epoch range: {epoch_numbers[0]} to {epoch_numbers[-1]}')
    
    # Clean up training data and history after each stage
    del history
    print(f'Stage {i+1} cleanup: training history deleted')
    
    # Monitor memory after training stage
    try:
        import psutil
        memory_usage = psutil.virtual_memory()
        print(f'System RAM after stage {i+1}: {memory_usage.used / 1024**3:.2f} GB used / {memory_usage.total / 1024**3:.2f} GB total ({memory_usage.percent:.1f}%)')
    except:
        pass

    if early_stop.stopped_epoch > 0:
        print(f'Early stopping triggered after {early_stop.stopped_epoch} epochs.')
    
    if i == len(inter_seps) - 1:
        print('>>> Reached the last interparticle separation. Ending training loop.')
        break
    
    # Save the model after each interparticle separation
    final_model_path = MODEL_PATH + MODEL_NAME + f'_L{inter_sep}.keras'
    final_weights_path = MODEL_PATH + MODEL_NAME + f'_L{inter_sep}_final.weights.h5'
    
    # Always save weights
    try:
        model.save_weights(final_weights_path)
        print(f'Model weights saved for interparticle separation L={inter_sep} Mpc/h.')
    except Exception as e:
        print(f'Warning: Failed to save weights: {e}')
    
    # Try to save full model
    try:
        model.save(final_model_path)
        print(f'Full model saved for interparticle separation L={inter_sep} Mpc/h.')
    except Exception as e:
        print(f'Warning: Failed to save full model: {e}, but weights were saved.')
    
    # Clean up training data to free memory for next iteration
    if i < len(inter_seps) - 1:  # Don't delete on last iteration
        # Only delete variables if they exist
        if 'train_dataset' in locals():
            del train_dataset
        if 'train_features' in locals():
            del train_features
        if 'train_labels' in locals():
            del train_labels
        import gc
        gc.collect()
        print('Training data cleaned up for memory management.')
    
    print('Proceeding to next interparticle separation...\n')
#================================================================
# Final evaluation on the validation set
#================================================================
print('Evaluating final model on validation set...')
if validation_strategy == 'target':
    print(f'Using target validation dataset: L={target_lambda} Mpc/h')
    eval_dataset = val_dataset
elif validation_strategy == 'stage':
    print(f'Using final stage validation dataset: L={inter_seps[-1]} Mpc/h')
    eval_dataset = val_dataset  # Already updated to final stage in the loop
elif validation_strategy == 'gradual':
    final_val_lambda = gradual_validation_map[inter_seps[-1]]
    print(f'Using final gradual validation dataset: L={final_val_lambda} Mpc/h')
    eval_dataset = val_dataset  # Already updated to final gradual stage in the loop
else:  # hybrid
    print(f'Using target validation dataset for final evaluation: L={target_lambda} Mpc/h')
    # For hybrid validation, target_val_dataset was created earlier
    eval_dataset = target_val_dataset

try:
    results = model.evaluate(eval_dataset, verbose=2)
    print('Final evaluation results:', results)
except Exception as e:
    print(f'Error during final evaluation: {e}')
    print('This may be due to memory constraints. Training completed successfully.')

# Clean up validation data and model to free memory before plotting
try:
    del val_dataset
    if validation_strategy == 'hybrid' and 'target_val_dataset' in locals():
        del target_val_dataset
    if validation_callback is not None:
        del validation_callback
except NameError:
    pass  # Variables may not exist
import gc
gc.collect()
print('Validation data cleaned up.')
#================================================================
# Plot training history
#================================================================
class CombinedHistory:
    def __init__(self, history_dict):
        self.history = history_dict
final_history = CombinedHistory(combined_history)
print('>>> Plotting training history...')
MODEL_FIG_PATH = FIG_PATH + MODEL_NAME + '/'
if not os.path.exists(MODEL_FIG_PATH):
    os.makedirs(MODEL_FIG_PATH)
FILE_METRICS = MODEL_FIG_PATH + MODEL_NAME + '_metrics.png'
plotter.plot_training_metrics_all(final_history, FILE_METRICS,savefig=True)
print(f'Training history plot saved to {FILE_METRICS}')
#================================================================
# Predictions on validation set and slice plots
#================================================================
'''
scores = {}
print('>>> Making predictions on validation set...')
predictions = model.predict(val_dataset, verbose=2, batch_size=BATCH_SIZE)
print('>>> Predictions made on validation set.')
print('Predictions shape:', predictions.shape)
# Calculate scores on validation set
FILE_PRED = PRED_PATH+ MODEL_NAME + f'_predictions_L{L_VAL}.fvol'
nets.save_scores_from_fvol(
    val_labels, predictions, MODEL_PATH + MODEL_NAME + f'_L{L_VAL}.h5',
    MODEL_FIG_PATH, scores, N_CLASSES, VAL_FLAG = True)
# Save slice plots:
nets.save_scores_from_model(DATA_PATH + data_info[L_VAL], FILE_MASK,
                           MODEL_PATH + MODEL_NAME + f'_L{L_VAL}.h5',
                           MODEL_FIG_PATH, FILE_PRED,
                           GRID=GRID, SUBGRID=SUBGRID, OFF=OFF,
                           TRAIN_SCORE=False, 
                           EXTRA_INPUTS=EXTRA_INPUTS_INFO[L_VAL] if EXTRA_INPUTS else None)
print('>>> Curricular training completed successfully.')
print('>>> Model and predictions saved in:', MODEL_FIG_PATH)
print('>>> Predictions saved in:', FILE_PRED)
print('>>> Training history plot saved in:', FILE_METRICS)
'''