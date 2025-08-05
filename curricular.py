#!/usr/bin/env python3
'''
7/28/25: Implementing an idea about curricular training. Instead of loading a model designed 
for a specific interparticle separation (hereto referrred to as lambda and in units of Mpc/h),
we will instead begin the training on the lowest interparticle separation and
progressively increase the interparticle separation.

You can either score the models on the highest lambda or on the current lambda. 
'''
print('>>> Running curricular.py')
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import NETS_LITE as nets
import absl.logging
import plotter
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
absl.logging.set_verbosity(absl.logging.ERROR)
print('TensorFlow version:', tf.__version__)
print('CUDA?', tf.test.is_built_with_cuda())
# get the GPU devices
gpus = tf.config.list_physical_devices('GPU')
print('GPUs available:', gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
nets.K.set_image_data_format('channels_last')
# NOTE turning off XLA JIT compilation for now, as it can cause issues with some models
tf.config.optimizer.set_jit(False)  # if you're using XLA
os.environ["TF_DISABLE_CUDNN_AUTOTUNE"] = "1"
tf.config.experimental.enable_op_determinism()
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
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
required.add_argument('LOSS', type=str, choices=['CCE', 'DISCCE', 'FOCAL_CCE', 'SCCE', 'SCCE_Void_Penalty', 'SCCE_Class_Penalty'],
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
optional.add_argument('--USE_ATTENTION', action='store_true',
                      help='Use attention U-Net architecture instead of standard U-Net.')
optional.add_argument('--LAMBDA_CONDITIONING', action='store_true',
                      help='Use lambda conditioning in the model.')
optional.add_argument('--N_EPOCHS_PER_INTER_SEP', type=int, default=50,
                      help='Number of epochs to train for each interparticle separation. Default is 50.')
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
print(f'Parsed arguments: ROOT_DIR={ROOT_DIR}, DEPTH={DEPTH}, FILTERS={FILTERS}, LOSS={LOSS}, UNIFORM_FLAG={UNIFORM_FLAG}, BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE}, LEARNING_RATE_PATIENCE={LEARNING_RATE_PATIENCE}, L_VAL={L_VAL}, USE_ATTENTION={USE_ATTENTION}, EXTRA_INPUTS={EXTRA_INPUTS}, ADD_RSD={ADD_RSD}, LAMBDA_CONDITIONING={LAMBDA_CONDITIONING}, N_EPOCHS_PER_INTER_SEP={N_EPOCHS_PER_INTER_SEP}')
# use mixed precision if on Picotte
if ROOT_DIR.startswith('/ifs/groups/vogeleyGrp/'):
    from tf.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
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
def load_data(inter_sep, extra_inputs=None):
    '''
    Load the data for a given interparticle separation.
    Args:
        inter_sep (str): Interparticle separation (lambda) as a string.
        extra_inputs (str, optional): Additional inputs file name. Defaults to None.
    Returns:
        features (np.ndarray): Features array.
        labels (np.ndarray): Labels array.
        extra_input (np.ndarray, optional): Additional inputs array if provided.
    '''
    if inter_sep not in inter_seps:
        raise ValueError(f'Invalid interparticle separation: {inter_sep}. Must be one of {inter_seps}.')
    data_file = DATA_PATH + data_info[inter_sep]
    print(f'Loading data from {data_file}...')
    features, labels = nets.load_dataset_all(
        FILE_DEN=data_file,
        FILE_MASK=FILE_MASK,
        SUBGRID=SUBGRID
    )
    if extra_inputs is not None:
        if inter_sep not in EXTRA_INPUTS_INFO:
            raise ValueError(f'Invalid interparticle separation for extra inputs: {inter_sep}. Must be one of {list(EXTRA_INPUTS_INFO.keys())}.')
        extra_input_file = DATA_PATH + EXTRA_INPUTS_INFO[inter_sep]
        print(f'Loading additional inputs from {extra_input_file}...')
        extra_input = nets.chunk_array(
            extra_input_file,
            SUBGRID=SUBGRID,
            scale=True
        )
        if extra_input.shape[:-1] != features.shape[:-1]:
            raise ValueError(f'Extra input shape {extra_input.shape} does not match features shape {features.shape}.')
        features = np.concatenate([features, extra_input], axis=-1)
    print(f'Features shape: {features.shape}, Labels shape: {labels.shape}')
    return features, labels
def make_dataset(delta, tij_labels, batch_size=BATCH_SIZE, shuffle=True, one_hot=False, lambda_value=None):
    '''
    Create a TensorFlow dataset from the features and labels.
    Args:
        delta (np.ndarray): Features array.
        tij_labels (np.ndarray): Labels array.
        batch_size (int, optional): Batch size. Defaults to BATCH_SIZE.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        one_hot (bool, optional): Whether to apply one-hot encoding to the labels. Defaults to True.
    Returns:
        tf.data.Dataset: A TensorFlow dataset.
    '''
    print(f'Creating dataset with {len(delta)} samples...')
    print(f'Features shape: {delta.shape}, Labels shape: {tij_labels.shape}')
    if one_hot:
        tij_labels = tf.keras.utils.to_categorical(tij_labels, num_classes=N_CLASSES)
        print(f'One-hot encoding applied. Labels shape: {tij_labels.shape}')
    # Check for NaN values in delta and tij_labels
    if np.any(np.isnan(delta)):
        raise ValueError('NaN values found in delta array.')
    if np.any(np.isnan(tij_labels)):
        raise ValueError('NaN values found in tij_labels array.')
    # Ensure delta is a float32 tensor
    delta = tf.convert_to_tensor(delta, dtype=tf.float32)
    # Ensure tij_labels is a float32 tensor
    tij_labels = tf.convert_to_tensor(tij_labels, dtype=tf.float32)
    # Create the dataset
    print(f'Creating dataset with batch size {batch_size} and shuffle={shuffle}...')
    if lambda_value is not None:
        print(f'Adding lambda input with value {lambda_value}')
        lambda_array = np.full((delta.shape[0], 1), lambda_value, dtype=np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((
            {'density_input': delta, 'lambda_input': lambda_array},
            {'last_activation': tij_labels, 'lambda_output': lambda_array}
        ))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((delta, tij_labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=delta.shape[0])
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print(f'Dataset created with {len(dataset)} batches.')
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
if USE_ATTENTION:
    MODEL_NAME += '_attention'
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
# Make fixed validation set (since we are interested in L=10 Mpc/h)
#================================================================
if 'SCCE' in LOSS or 'DISCCE' in LOSS:
    ONE_HOT = False
else:
    ONE_HOT = True
print(f'ONE_HOT encoding: {ONE_HOT}')
print(f'Loading validation data for interparticle separation L={L_VAL} Mpc/h...')
if L_VAL not in inter_seps:
    raise ValueError(f'Invalid interparticle separation for validation: {L_VAL}. Must be one of {inter_seps}.')
if EXTRA_INPUTS:
    val_features, val_labels = load_data(L_VAL, extra_inputs=EXTRA_INPUTS)
else:
    val_features, val_labels = load_data(L_VAL)
val_dataset = make_dataset(val_features, val_labels, batch_size=BATCH_SIZE, shuffle=False, one_hot=ONE_HOT, lambda_value=float(L_VAL) if LAMBDA_CONDITIONING else None)
#================================================================
# Set loss function and metrics
#================================================================
metrics = ['accuracy']
metrics += [nets.MCC_keras(int_labels=not ONE_HOT),
            nets.F1_micro_keras(int_labels=not ONE_HOT),
            nets.void_F1_keras(int_labels=not ONE_HOT)]
if LOSS == 'CCE':
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
elif LOSS == 'DISCCE':
    loss_fn = [nets.SCCE_Dice_loss]
elif LOSS == 'FOCAL_CCE':
    alpha = [0.4, 0.4, 0.15, 0.05]
    gamma = 2.0
    loss_fn = [nets.categorical_focal_loss(alpha=alpha, gamma=gamma)]
    print(f'Using Focal Loss with alpha={alpha} and gamma={gamma}')
elif LOSS == 'SCCE':
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
elif LOSS == 'SCCE_Void_Penalty':
    loss_fn = [nets.SCCE_void_penalty]
elif LOSS == 'SCCE_Class_Penalty':
    #target_props = [0.65, 0.26, 0.09, 0.005]  # Example target proportions for void, wall, filament, halo
    target_props = None
    penalty_weights = [1.0, 0.9, 0.3, 0.1]
    penalty_type = 'mse'  # Mean Squared Error for penalty
    loss_fn = lambda y_true, y_pred: nets.SCCE_class_proportion_penalty(
        y_true, y_pred, target_proportions=target_props,
        weights=penalty_weights, penalty_type=penalty_type
    )
# Make tensorboard directory
log_dir = ROOT_DIR + 'logs/fit/' + MODEL_NAME + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '/'
os.makedirs(log_dir, exist_ok=True)
#================================================================
# Create the model
#================================================================
print(f'Creating model with depth={DEPTH}, filters={FILTERS}, loss={LOSS}, uniform={UNIFORM_FLAG}, RSD={ADD_RSD}, attention={USE_ATTENTION}...')
#strategy = tf.distribute.MirroredStrategy()
strategy = tf.distribute.get_strategy()
print('Number of devices:', strategy.num_replicas_in_sync)
with strategy.scope():
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
    if LAMBDA_CONDITIONING:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE,clipnorm=1.0),  # Gradient clipping to prevent exploding gradients
            loss={'last_activation': loss_fn, 'lambda_output': 'mse'},
            loss_weights={'last_activation': 1.0, 'lambda_output': 0.1},
            metrics={'last_activation': metrics, 'lambda_output': 'mse'}
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=LEARNING_RATE,
                clipnorm=1.0  # Gradient clipping to prevent exploding gradients
                ),
            loss=loss_fn,
            metrics=metrics
        )
print(model.summary())
#================================================================
# Training loop
#================================================================
print('>>> Starting curricular training...')
reduce_LR = ReduceLROnPlateau(
            patience=LEARNING_RATE_PATIENCE,
            factor=0.5,
            monitor='val_loss',
            mode='min',
            verbose=1,
            min_lr=1e-6
)
# set freezing scheme:
density_to_freeze_map = {
    '0.33': 0,  # No freezing for the lowest interparticle separation
    '3': 0,     # Freeze first block for L=3 Mpc/h
    '5': 1,     # Freeze first two blocks for L=5 Mpc/h
    '7': 2,     # Freeze first three blocks for L=7 Mpc/h
    '10': 3     # Freeze first four blocks for L=10 Mpc/h
}
# create combined history object to store metrics for all interparticle separations
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
tensor_board_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=False,
    update_freq='epoch',
)
#================================================================
# Training loop over interparticle separations
#================================================================
print('>>> Starting training loop over interparticle separations...')
epoch_offset = 0
for i, inter_sep in enumerate(inter_seps):
    print(f'Starting training for interparticle separation L={inter_sep} Mpc/h...')
    if EXTRA_INPUTS is not None:
        train_features, train_labels = load_data(inter_sep, extra_inputs=EXTRA_INPUTS)
    else:
        train_features, train_labels = load_data(inter_sep)
    print(f'Training data loaded for L={inter_sep}. Features shape: {train_features.shape}, Labels shape: {train_labels.shape}')
    # Create the training dataset
    train_dataset = make_dataset(train_features, train_labels, batch_size=BATCH_SIZE, shuffle=True, one_hot=ONE_HOT, lambda_value=float(inter_sep) if LAMBDA_CONDITIONING else None)
    # freeze layers based on interparticle separation. freeze more layers for larger interparticle separations:
    nets.freeze_encoder_blocks(
        model,
        density_to_freeze_map[inter_sep],
    )
    if inter_sep != '0.33':
        # recompile the model after freezing layers
        if LAMBDA_CONDITIONING:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss={'last_activation': loss_fn, 'lambda_output': 'mse'},
                loss_weights={'last_activation': 1.0, 'lambda_output': 0.1},
                metrics={'last_activation': metrics, 'lambda_output': 'mse'})
        else:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
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
    callbacks = [
        ModelCheckpoint(
            filepath=checkpt_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),
        tensor_board_callback,
        reduce_LR,
        early_stop,
        void_fraction_monitor
    ]
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

    # always load the best model weights after training
    if os.path.exists(checkpt_path):
        print(f'Loading best model from {checkpt_path}')
        best_model = tf.keras.models.load_model(checkpt_path, custom_objects={
            'MCC_keras': nets.MCC_keras,
            'F1_micro_keras': nets.F1_micro_keras,
            'void_F1_keras': nets.void_F1_keras,
            'SCCE_Dice_loss': nets.SCCE_Dice_loss,
            'categorical_focal_loss': nets.categorical_focal_loss,
            'SCCE_void_penalty': nets.SCCE_void_penalty,
            'categorical_focal_loss': nets.categorical_focal_loss,
            'VoidFractionMonitor': nets.VoidFractionMonitor,
            'Cast': tf.keras.layers.Lambda
        },
        compile=False)
        model.set_weights(best_model.get_weights())
        print('Best model weights loaded.')

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
    

    if early_stop.stopped_epoch > 0:
        print(f'Early stopping triggered after {early_stop.stopped_epoch} epochs.')
    
    if i == len(inter_seps) - 1:
        print('>>> Reached the last interparticle separation. Ending training loop.')
        break
    
    # Save the model after each interparticle separation
    model.save(MODEL_PATH + MODEL_NAME + f'_L{inter_sep}.h5')
    print(f'Model saved for interparticle separation L={inter_sep} Mpc/h.')
    print('Proceeding to next interparticle separation...\n')
#================================================================
# Final evaluation on the validation set
#================================================================
print('Evaluating final model on validation set...')
results = model.evaluate(val_dataset, verbose=2)
print('Final evaluation results:', results)
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