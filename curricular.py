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
nets.K.set_image_data_format('channels_last')
#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')
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
required.add_argument('LOSS', type=str, choices=['CCE', 'DISCCE', 'FOCAL_CCE', 'SCCE', 'SCCE_Void_Penalty'],
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
                      help='Additional inputs for the model such as color or fluxes.')
optional.add_argument('--ADD_RSD', action='store_true',
                      help='Add RSD (Redshift Space Distortion) to the inputs.')
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
print(f'Parsed arguments: ROOT_DIR={ROOT_DIR}, DEPTH={DEPTH}, FILTERS={FILTERS}, LOSS={LOSS}, UNIFORM_FLAG={UNIFORM_FLAG}, BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE}, LEARNING_RATE_PATIENCE={LEARNING_RATE_PATIENCE}, EXTRA_INPUTS={EXTRA_INPUTS}, ADD_RSD={ADD_RSD}')
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
        print(f'Loading additional inputs from {extra_inputs}...')
        extra_data_file = DATA_PATH + extra_inputs
        extra_input = nets.chunk_array(
            extra_data_file,
            SUBGRID=SUBGRID,
            scale=True
        )
    print(f'Features shape: {features.shape}, Labels shape: {labels.shape}')
    return features, labels, extra_input if extra_inputs else None
def make_dataset(delta, tij_labels, batch_size=BATCH_SIZE, shuffle=True, one_hot=False):
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
    dataset = tf.data.Dataset.from_tensor_slices((delta, tij_labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(delta))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
#================================================================
# Make fixed validation set (since we are interested in L=10 Mpc/h)
#================================================================
print('Loading validation data for interparticle separation L=10 Mpc/h...')
L_VAL = '10'
if L_VAL not in inter_seps:
    raise ValueError(f'Invalid interparticle separation for validation: {L_VAL}. Must be one of {inter_seps}.')
if EXTRA_INPUTS:
    val_features, val_labels, val_extra_input = load_data(L_VAL, extra_inputs=EXTRA_INPUTS)
else:
    val_features, val_labels, val_extra_input = load_data(L_VAL)
val_dataset = make_dataset(val_features, val_labels, batch_size=BATCH_SIZE, shuffle=False)
if EXTRA_INPUTS:
    val_extra_input = tf.data.Dataset.from_tensor_slices(val_extra_input).batch(BATCH_SIZE)
#================================================================
# Create model parameters
#================================================================
SIM = 'TNG'
MODEL_NAME = f'{SIM}_curricular_{LOSS}_D{DEPTH}_F{FILTERS}'
if UNIFORM_FLAG:
    MODEL_NAME += '_uniform'
if ADD_RSD:
    MODEL_NAME += '_RSD'
if EXTRA_INPUTS:
    pass # Add logic for handling extra inputs later
DATE = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
MODEL_NAME += f'_{DATE}'
print(f'Model name: {MODEL_NAME}')
last_activation = 'softmax' # NOTE implement changing later 
input_shape = (None, None, None, 1)
if EXTRA_INPUTS:
    input_shape = (None, None, None, 1 + len(EXTRA_INPUTS.split(','))) # Adjust for extra inputs
print(f'Input shape: {input_shape}')
#================================================================
# Set loss function and metrics
#================================================================
if 'SCCE' in LOSS or 'DISCCE' in LOSS:
    ONE_HOT = False
else:
    ONE_HOT = True
print(f'ONE_HOT encoding: {ONE_HOT}')
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
# Make tensorboard directory
log_dir = ROOT_DIR + 'logs/fit/' + MODEL_NAME + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '/'
os.makedirs(log_dir, exist_ok=True)
#================================================================
# Create the model
#================================================================
print(f'Creating model with depth={DEPTH}, filters={FILTERS}, loss={LOSS}, uniform={UNIFORM_FLAG}, RSD={ADD_RSD}...')
#strategy = tf.distribute.MirroredStrategy()
strategy = tf.distribute.get_strategy()
print('Number of devices:', strategy.num_replicas_in_sync)
with strategy.scope():
    model = nets.unet_3d(
        input_shape=input_shape,
        num_classes=N_CLASSES,
        initial_filters=FILTERS,
        depth=DEPTH,
        last_activation=last_activation,
        batch_normalization=True,
        model_name=MODEL_NAME
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=metrics
    )
print(model.summary())
#================================================================
# Training loop
#================================================================
print('>>> Starting curricular training...')
N_EPOCHS_PER_INTER_SEP = 50  # Number of epochs per interparticle separation
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
    'val_void_fraction': []
}
#================================================================
# Training loop over interparticle separations
#================================================================
print('>>> Starting training loop over interparticle separations...')
for i, inter_sep in enumerate(inter_seps):
    print(f'Starting training for interparticle separation L={inter_sep} Mpc/h...')
    if EXTRA_INPUTS:
        train_features, train_labels, train_extra_input = load_data(inter_sep, extra_inputs=EXTRA_INPUTS)
    else:
        train_features, train_labels, train_extra_input = load_data(inter_sep)
    
    # Create the training dataset
    train_dataset = make_dataset(train_features, train_labels, batch_size=BATCH_SIZE, shuffle=True, one_hot=ONE_HOT)
    # freeze layers based on interparticle separation. freeze more layers for larger interparticle separations:
    nets.freeze_encoder_blocks(
        model,
        density_to_freeze_map[inter_sep],
    )
    if inter_sep != '0.33':
        # recompile the model after freezing layers
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=loss_fn,
            metrics=metrics
    )
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
    callbacks = [
        ModelCheckpoint(
            filepath=MODEL_PATH + MODEL_NAME + f'_L{inter_sep}.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=False,
            update_freq='epoch',
        ),
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

    # Append the history to the combined history
    for key in combined_history.keys():
        if key in history.history:
            combined_history[key].extend(history.history[key])
    print(f'Combined history now has {len(combined_history["loss"])} total epochs')
    

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
scores = {}
print('>>> Making predictions on validation set...')
predictions = model.predict(val_dataset, verbose=2, batch_size=BATCH_SIZE)
if EXTRA_INPUTS:
    val_extra_input = tf.concat([val_features, val_extra_input], axis=-1)
FILE_PRED = MODEL_FIG_PATH + 'predictions_L10.png'
nets.save_scores_from_fvol(
    val_labels, predictions, MODEL_PATH + MODEL_NAME + '_L10.h5',
    MODEL_FIG_PATH, scores, N_CLASSES, VAL_FLAG = True)
# Save slice plots:
nets.save_scores_from_model(DATA_PATH + data_info['10'], FILE_MASK,
                           MODEL_PATH + MODEL_NAME + '_L10.h5',
                           MODEL_FIG_PATH, FILE_PRED,
                           GRID=GRID, SUBGRID=SUBGRID, OFF=OFF,
                           TRAIN_SCORE=False, EXTRA_INPUTS=EXTRA_INPUTS)
print('>>> Curricular training completed successfully.')
print('>>> Model and predictions saved in:', MODEL_FIG_PATH)
print('>>> Predictions saved in:', FILE_PRED)
print('>>> Training history plot saved in:', FILE_METRICS)