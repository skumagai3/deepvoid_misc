#!/usr/bin/env python3
'''
3/17/24: Making an updated version of dv-train-nonbinary.py in nets.
I want to also make a lighter-weight version of the nets.py file,
which currently imports all kinds of stuff that I don't need.
'''
print('>>> Running DV_MULTI_TRAIN.py')
import os
import sys
import argparse
import datetime
import numpy as np
import tensorflow as tf
import NETS_LITE as nets
import absl.logging
import plotter
absl.logging.set_verbosity(absl.logging.ERROR)
print('TensorFlow version:', tf.__version__)
#print('Keras version:', tf.keras.__version__)
print('CUDA?:',tf.test.is_built_with_cuda())
nets.K.set_image_data_format('channels_last')
# only use with Nvidia GPUs with compute capability >= 7.0!
#from tensorflow.keras import mixed_precision  # type: ignore
#mixed_precision.set_global_policy('mixed_float16')
#===============================================================
# Set training parameters:
#===============================================================
#epochs = 500; print('epochs:',epochs)
#patience = 25; print('patience:',patience)
#lr_patience = 10; print('learning rate patience:',lr_patience)
N_epochs_metric = 10
print(f'classification metrics calculated every {N_epochs_metric} epochs')
KERNEL = (3,3,3)
#===============================================================
# Set random seed
#===============================================================
seed = 12; print('Setting random seed:',seed)
np.random.seed(seed)
tf.random.set_seed(seed)
#===============================================================
# Set parameters and paths
#===============================================================
class_labels = ['void','wall','fila','halo']
N_CLASSES = 4
#===============================================================
# arg parsing:
#===============================================================
'''
Usage: python3 DV_MULTI_TRAIN.py <ROOT_DIR> <SIM> <L> <DEPTH> <FILTERS> <LOSS> <GRID> [--UNIFORM_FLAG] [--BATCHNORM] [--DROPOUT] [--MULTI_FLAG] [--LOW_MEM_FLAG]

Arguments:
  ROOT_DIR: Root directory where models, predictions, and figures will be saved.
  SIM: Simulation type. Either 'TNG' or 'BOL'.
  L: Interparticle separation in Mpc/h. 
    For TNG full DM use '0.33', for BOL full DM use '0.122'. 
    Other valid values are '1', '3', '5', '7', '10'.
  DEPTH: Depth of the U-Net. Default is 3.
  FILTERS: Number of filters in the first layer. Default is 32.
  LOSS: Loss function to use. Options are 'CCE', 'SCCE', 'FOCAL_CCE'. Default is 'CCE'.
  GRID: Size of the density and mask fields on a side. For TNG, use 512, for Bolshoi, use 640.

Optional Flags:
  --UNIFORM_FLAG: If set, use identical masses for all subhaloes. Default is False.
  --BATCHNORM: If set, use batch normalization. Default is False.
  --DROPOUT: Dropout rate. Default is 0.0 (no dropout).
  --MULTI_FLAG: If set, use multiprocessing. Default is False.
  --LOW_MEM_FLAG: If set, will load less training data and do not report metrics. Default is True.
  --FOCAL_ALPHA: Focal loss alpha parameter. Default is 0.25. can be a list of 4 values.
  --FOCAL_GAMMA: Focal loss gamma parameter. Default is 2.0.
  --MODEL_NAME_SUFFIX: Suffix to add to model name. Default is empty.
  --LOAD_MODEL_FLAG: If set, load model from FILE_OUT if it exists. Default is False.
  --LOAD_INTO_MEM: If set, load training and test data into memory. 
    If not set, load data from X_train, Y_train, X_test, Y_test .npy files into a tf.data.Dataset 
    object that will load the data in batches instead of all at once. Default is False.
  --BATCH_SIZE: Batch size. Default is 4.
  --EPOCHS: Number of epochs to train for. Default is 500.
  --LEARNING_RATE: Initial learning rate. Default is 0.001.
  --LEARNING_RATE_PATIENCE: Number of epochs to wait before reducing learning rate. Default is 10.
  --REGULARIZE_FLAG: If set, use L2 regularization. Default is False.
  --PICOTTE_FLAG: If set, set up for sbatch run on Picotte. Default is False. 
    So far all this flag will do is disable the extra metrics since Picotte is on
    an older version of TensorFlow/Keras (2.10.1, while Colab is on 2.15.0).
  --TENSORBOARD_FLAG: If set, use tensorboard. Default is False.
'''
parser = argparse.ArgumentParser(
  prog='DV_MULTI_TRAIN.py',
  description='Train a U-Net on a multiclass morphological mask')
# required args: ROOT_DIR, SIM, L, DEPTH, FILTERS, LOSS, GRID
req_group = parser.add_argument_group('required arguments')
req_group.add_argument('ROOT_DIR', type=str, help='Root directory where models, predictions, and figures will be saved.')
req_group.add_argument('SIM', type=str, help='Simulation: TNG or BOL.')
req_group.add_argument('L', type=lambda x: float(x) if x in ['0.33', '0.122'] else int(x), 
                       help='Interparticle separation in Mpc/h. TNG full DM 0.33, BOL full DM 0.122, 3,5,7,10.')
req_group.add_argument('DEPTH', type=int, default=3, help='Depth of the U-Net.')
req_group.add_argument('FILTERS', type=int, default=32, help='Number of filters in the first layer.')
req_group.add_argument('LOSS', type=str, default='CCE', help='Loss function to use: CCE, SCCE, FOCAL_CCE, DISCCE, SCCE_Void_Penalty.')
req_group.add_argument('GRID', type=int, help='Size of the density and mask fields on a side. For TNG, GRID=512, for Bolshoi, GRID=640.')
# optional args: UNIFORM_FLAG, BATCHNORM, DROPOUT, MULTI_FLAG, LOW_MEM_FLAG
opt_group = parser.add_argument_group('optional arguments')
opt_group.add_argument('--UNIFORM_FLAG', action='store_true', help='If set, use identical masses for all subhaloes.')
opt_group.add_argument('--BATCHNORM', action='store_true', help='If set, use batch normalization')
opt_group.add_argument('--DROPOUT', type=float, default=0.0, help='Dropout rate. 0.0 means no dropout.')
opt_group.add_argument('--MULTI_FLAG', action='store_true', help='If set, use multiprocessing.')
opt_group.add_argument('--LOW_MEM_FLAG', action='store_false', help='If not set, will load less training data and report less metrics.')
opt_group.add_argument('--FOCAL_ALPHA', type=float, nargs='+', default=[0.25,0.25,0.25,0.25], help='Focal loss alpha parameter. Default is 0.25.')
opt_group.add_argument('--FOCAL_GAMMA', type=float, default=2.0, help='Focal loss gamma parameter. Default is 2.0.')
opt_group.add_argument('--MODEL_NAME_SUFFIX', type=str, default='', help='Suffix to add to model name. Default is empty.')
opt_group.add_argument('--LOAD_MODEL_FLAG', action='store_true', help='If set, load model from FILE_OUT if it exists.')
opt_group.add_argument('--LOAD_INTO_MEM', action='store_true', help='If set, load all training and test data into memory. Default is False, aka to load from train, test .npy files into a tf.data.Dataset object.')
opt_group.add_argument('--BATCH_SIZE', type=int, default=8, help='Batch size. Default is 4.')
opt_group.add_argument('--EPOCHS', type=int, default=500, help='Number of epochs to train for. Default is 500.')
opt_group.add_argument('--LEARNING_RATE', type=float, default=0.001, help='Initial learning rate. Default is 3e-3.')
opt_group.add_argument('--LEARNING_RATE_PATIENCE', type=int, default=10, help='Number of epochs to wait before reducing learning rate. Default is 10.')
opt_group.add_argument('--PATIENCE', type=int, default=25, help='Number of epochs to wait before early stopping. Default is 25.')
opt_group.add_argument('--REGULARIZE_FLAG', action='store_true', help='If set, use L2 regularization.')
opt_group.add_argument('--PICOTTE_FLAG', action='store_true', help='If set, set up for sbatch run on Picotte.')
opt_group.add_argument('--TENSORBOARD_FLAG', action='store_true', help='If set, use tensorboard.')
opt_group.add_argument('--BINARY_MASK', action='store_true', help='If set, use binary mask.')
opt_group.add_argument('--BOUNDARY_MASK', type=str, default=None, help='If set, model training does not count out of bounds voxels in the loss calculation.')
opt_group.add_argument('--EXTRA_INPUTS', type=str, default=None, help='If set, use extra inputs for the model. Should be a filename of a .fvol file.')
opt_group.add_argument('--ADD_RSD', action='store_true', help='If set, add RSD perturbation to the density field. This is only for TNG300-3-Dark (so far).')
opt_group.add_argument('--USE_PCONV', action='store_true', help='If set, use PartialConv3D instead of Conv3D.')
args = parser.parse_args()
ROOT_DIR = args.ROOT_DIR
SIM = args.SIM
L = args.L
DEPTH = args.DEPTH
FILTERS = args.FILTERS
LOSS = args.LOSS
GRID = args.GRID
UNIFORM_FLAG = args.UNIFORM_FLAG
BATCHNORM = args.BATCHNORM
DROPOUT = args.DROPOUT
MULTI_FLAG = args.MULTI_FLAG
LOW_MEM_FLAG = args.LOW_MEM_FLAG
alpha = args.FOCAL_ALPHA
gamma = args.FOCAL_GAMMA
MODEL_NAME_SUFFIX = args.MODEL_NAME_SUFFIX
LOAD_MODEL_FLAG = args.LOAD_MODEL_FLAG
LOAD_INTO_MEM = args.LOAD_INTO_MEM
batch_size = args.BATCH_SIZE
epochs = args.EPOCHS
LR = args.LEARNING_RATE
LR_PATIENCE = args.LEARNING_RATE_PATIENCE
PATIENCE = args.PATIENCE
REGULARIZE_FLAG = args.REGULARIZE_FLAG
PICOTTE_FLAG = args.PICOTTE_FLAG
TENSORBOARD_FLAG = args.TENSORBOARD_FLAG
BINARY_MASK = args.BINARY_MASK
BOUNDARY_MASK = args.BOUNDARY_MASK
EXTRA_INPUTS = args.EXTRA_INPUTS
ADD_RSD = args.ADD_RSD
USE_PCONV = args.USE_PCONV
print('#############################################')
print('#############################################')
print('>>> Running DV_MULTI_TRAIN.py')
print('>>> Root directory:',ROOT_DIR)
print('>>> Parameters:')
print('Simulation =', SIM); 
print('L =',L); 
print('DEPTH =',DEPTH); print('FILTERS =',FILTERS)
print('GRID =',GRID)
print(f'Batch normalization: {bool(BATCHNORM)}')
print('DROPOUT =',DROPOUT)
print(f'L2 Regularization: {bool(REGULARIZE_FLAG)}')
print('LEARNING_RATE =',LR)
print('LR_PATIENCE =',LR_PATIENCE)
print('LOSS =',LOSS)
if LOSS == 'FOCAL_CCE':
  print('FOCAL_ALPHA =',alpha)
  print('FOCAL_GAMMA =',gamma)
print('MODEL_NAME_SUFFIX =',MODEL_NAME_SUFFIX)
print('BATCH_SIZE =',batch_size)
print('EPOCHS =',epochs)
print(f'Use uniform subhalo masses: {bool(UNIFORM_FLAG)}')
print(f'Load existing model flag: {bool(LOAD_MODEL_FLAG)}')
print(f'Load entire dataset into memory: {bool(LOAD_INTO_MEM)}')
print(f'Use saved memory-mapped arrays: {bool(not LOAD_INTO_MEM)}')
print(f'Low memory flag: {bool(LOW_MEM_FLAG)}')
print(f'Multiprocessing: {bool(MULTI_FLAG)}')
print(f'Picotte flag: {bool(PICOTTE_FLAG)}')
print(f'Tensorboard: {bool(TENSORBOARD_FLAG)}')
print(f'Using binary mask: {bool(BINARY_MASK)}')
if BOUNDARY_MASK is not None:
  print(f'Using boundary mask: {BOUNDARY_MASK}')
if EXTRA_INPUTS is not None:
  print(f'Using extra input: {EXTRA_INPUTS}')
  N_CHANNELS = 2 # density + color fields
if ADD_RSD:
  print('>>> Adding RSD perturbation to the density field')
  if SIM != 'TNG':
    sys.exit('ERROR: RSD perturbation only implemented for TNG300 base density (so far).') #NOTE
if USE_PCONV:
  print('>>> Using PartialConv3D instead of Conv3D')
  if EXTRA_INPUTS is None:
    print('>>> Using PartialConv3D without any mask input')
print('#############################################')
print('#############################################')
#===============================================================
# set paths
#===============================================================
path_to_TNG = ROOT_DIR + 'data/TNG/'
path_to_BOL = ROOT_DIR + 'data/Bolshoi/'
FIG_DIR_PATH = ROOT_DIR + 'figs/'
FILE_OUT = ROOT_DIR + 'models/'
FILE_PRED = ROOT_DIR + 'preds/'
if not os.path.exists(FIG_DIR_PATH):
  os.makedirs(FIG_DIR_PATH)
if not os.path.exists(FILE_OUT):
  os.makedirs(FILE_OUT)
if not os.path.exists(FILE_PRED):
  os.makedirs(FILE_PRED)
#===============================================================
# Set filenames based on simulation, L, and UNIFORM_FLAG.
# FILE_DEN is the density field, FILE_MASK is the mask field.
# FILENAMES!!!
### TNG ### 
# full DM density [L=0.33 Mpc/h]: DM_DEN_snap99_Nm=512.fvol
# subhalo density: subs1_mass_Nm512_L{L}_d_None_smooth.fvol
# uniform subhalo density: subs1_mass_Nm512_L{L}_d_None_smooth_uniform.fvol
# mask: TNG300-3-Dark-mask-Nm=512-th=0.65-sig=2.4.fvol
# TNG Lambdas: 1, 3, 5, 7, 10, 12, 15

### Bolshoi ###
# full DM density [L=0.122 Mpc/h]: Bolshoi_halo_CIC_640_L=0.122.fvol
# subhalo density: Bolshoi_halo_CIC_640_L={L}.0.fvol
# mask: Bolshoi_bolshoi.delta416_mask_Nm=640_sig=0.916_thresh=0.65.fvol
# Bolshoi Lambdas: 2, 3, 5, 7, 10, 15
#===============================================================
th = 0.65 # eigenvalue threshold NOTE SET THRESHOLD HERE
# fix SIM prefix name for Bolshoi if provided as 'Bolshoi'
if SIM == 'Bolshoi':
  SIM = 'BOL'
  if BINARY_MASK:
    sys.exit('ERROR: BINARY_MASK not implemented for Bolshoi')
# set up .npy filepaths for saving/loading data
X_PREFIX = f'{SIM}_L{L}_Nm={GRID}'
Y_PREFIX = f'{SIM}_Nm={GRID}'
if LOSS == 'SCCE' or LOSS == 'DISCCE':
  Y_PREFIX += '_int'
  if BINARY_MASK:
    sys.exit('ERROR: categorical crossentropy loss not compatible with binary mask')
if BINARY_MASK:
  N_CLASSES = 2
  class_labels = ['void','wall']
### TNG ### 
if SIM == 'TNG':
  BoxSize = 205.0 # Mpc/h
  #GRID = 512 
  SUBGRID = 128
  OFF = 64
  ### TNG Density field:
  # FULL (L=0.33 Mpc/h)
  FILE_DEN_FULL = path_to_TNG + f'DM_DEN_snap99_Nm={GRID}.fvol'
  if ADD_RSD and L == 0.33:
    print('>>> Using RSD perturbed density field')
    FILE_DEN_FULL = path_to_TNG + 'DM_DEN_snap99_perturbed_Nm=512.fvol'
  if UNIFORM_FLAG == True:
    FILE_DEN_SUBS = path_to_TNG + f'subs1_mass_Nm{GRID}_L{L}_d_None_smooth_uniform.fvol'
  if UNIFORM_FLAG == False:
    FILE_DEN_SUBS = path_to_TNG + f'subs1_mass_Nm{GRID}_L{L}_d_None_smooth.fvol'
  if ADD_RSD and L > 0.33:
    # as of 7/21/25, there are no uniform mass RSD perturbed subhalo density fields
    print('>>> Using RSD perturbed subhalo density field')
    FILE_DEN_SUBS = path_to_TNG + f'subs1_mass_Nm{int(GRID)}_L{L}_RSD.fvol'
  if L == 0.33:
    FILE_DEN = FILE_DEN_FULL
    L = 0.33 # full DM interparticle separation for TNG300-3-Dark
    UNIFORM_FLAG = False # doesnt affect full DM, so set to False
  else:
    FILE_DEN = FILE_DEN_SUBS
  ### Mask field:
  sig = 2.4 # PHI smooothing scale in code units NOTE CHANGES WITH NM
  if GRID == 128:
    sig = 0.6
    SUBGRID = 32; OFF = 16
  if GRID == 256:
    sig = 1.2
    SUBGRID = 64; OFF = 32
  FILE_MASK = path_to_TNG + f'TNG300-3-Dark-mask-Nm={GRID}-th={th}-sig={sig}.fvol'
  FILE_FIG = FIG_DIR_PATH + 'TNG/'
  if not os.path.exists(FILE_FIG):
    os.makedirs(FILE_FIG)
  # set .npy filepaths up for saving if LOAD_INTO_MEM = True, and loading if LOAD_INTO_MEM = False
  FILE_X_TRAIN = path_to_TNG + X_PREFIX + '_X_train.npy'
  FILE_Y_TRAIN = path_to_TNG + Y_PREFIX + '_Y_train.npy'
  FILE_X_TEST  = path_to_TNG + X_PREFIX + '_X_test.npy'
  FILE_Y_TEST  = path_to_TNG + Y_PREFIX + '_Y_test.npy'
  if BINARY_MASK:
    FILE_Y_TRAIN = path_to_TNG + Y_PREFIX + '_Y_train_binary.npy'
    FILE_Y_TEST  = path_to_TNG + Y_PREFIX + '_Y_test_binary.npy'
### Bolshoi ###
elif SIM == 'BOL' or SIM == 'Bolshoi':
  SIM = 'BOL'
  #GRID = 640
  SUBGRID = 128
  OFF = 64
  UNIFORM_FLAG = False # don't have this for Bolshoi, so set to False
  if L == 0.122:
    FILE_DEN = path_to_BOL + f'Bolshoi_halo_CIC_{GRID}_L=0.122.fvol'
  else:
    FILE_DEN = path_to_BOL + f'Bolshoi_halo_CIC_{GRID}_L={L}.0.fvol'
  ### Mask field:
  sig = 0.916 # PHI smooothing scale in code units NOTE CHANGES WITH NM
  FILE_MASK = path_to_BOL + f'Bolshoi_bolshoi.delta416_mask_Nm={GRID}_sig={sig}_thresh={th}.fvol'
  FILE_FIG = FIG_DIR_PATH + 'Bolshoi/'
  if not os.path.exists(FILE_FIG):
    os.makedirs(FILE_FIG)
  # set .npy filepaths up for saving if LOAD_INTO_MEM = True, and loading if LOAD_INTO_MEM = False
  FILE_X_TRAIN = path_to_BOL + X_PREFIX + '_X_train.npy'
  FILE_Y_TRAIN = path_to_BOL + Y_PREFIX + '_Y_train.npy'
  FILE_X_TEST  = path_to_BOL + X_PREFIX + '_X_test.npy'
  FILE_Y_TEST  = path_to_BOL + Y_PREFIX + '_Y_test.npy'
# if L = 0.33 and SIM = TNG, error out, same for L = 0.122 and SIM = BOL:
if L == 0.33 and SIM == 'BOL':
  print('ERROR: L = 0.33 and SIM = BOL. L for full DM Bolshoi is 0.122.')
  sys.exit()
if L == 0.122 and SIM == 'TNG':
  print('ERROR: L = 0.122 and SIM = TNG. L for full DM TNG300-3-Dark is 0.33.')
  sys.exit()
#===============================================================
# Load data, normalize
#===============================================================
print('>>> Loading data!')
print('Density field:',FILE_DEN)
print('Mask field:',FILE_MASK)
if LOAD_INTO_MEM:
  print('>>> Loading full train, val data into memory')
  if LOW_MEM_FLAG:
    try: 
      features, labels = nets.load_dataset_all(FILE_DEN,FILE_MASK,SUBGRID)
    except FileNotFoundError:
      print('>>> File not found, trying with _RSD suffix')
      features, labels = nets.load_dataset_all(FILE_DEN.replace('.fvol','_RSD.fvol'),FILE_MASK,SUBGRID)
  else:
    features, labels = nets.load_dataset_all_overlap(FILE_DEN,FILE_MASK,SUBGRID,OFF)
  # CHECK FOR NANs
  if np.isnan(features).any():
    print('>>> WARNING: NaNs found in features, replacing with 0')
    features = np.nan_to_num(features)
  if np.isnan(labels).any():
    print('>>> WARNING: NaNs found in labels, replacing with 0')
    labels = np.nan_to_num(labels)
  print('>>> Data loaded!'); print('Features shape:',features.shape)
  print('Labels shape:',labels.shape)
  if BOUNDARY_MASK is not None:
    print('>>> Loading boundary mask:',BOUNDARY_MASK)
    boundary = nets.chunk_array(BOUNDARY_MASK, SUBGRID)
    print('>>> Boundary mask loaded!'); print('Boundary shape:',boundary.shape)
  if EXTRA_INPUTS is not None:
    print('>>> Loading extra inputs:',EXTRA_INPUTS)
    extra_input = nets.chunk_array(EXTRA_INPUTS, SUBGRID, scale=True)
    print('>>> Extra inputs loaded!'); print('Extra input shape:',extra_input.shape)
  # split into training and validation sets:
  # X_train is the density subcubes used to train the model
  # Y_train is the corresponding mask subcubes
  # X_test is the density subcubes used to validate the model
  # Y_test is the corresponding mask subcubes
  test_size = 0.2
  X_index = np.arange(0, features.shape[0])
  X_train, X_test, Y_train, Y_test = nets.train_test_split(X_index,labels,
                                                          test_size=test_size,
                                                          random_state=seed)
  if BOUNDARY_MASK is not None:
    boundary_train = boundary[X_train]; boundary_test = boundary[X_test]
    print('>>> Boundary mask split into training and test sets')
    print('Boundary train shape:',boundary_train.shape)
    print('Boundary test shape:',boundary_test.shape)
  if EXTRA_INPUTS is not None:
    extra_train = extra_input[X_train]; extra_test = extra_input[X_test]
    print('>>> Extra inputs split into training and test sets')
    print('Extra train shape:',extra_train.shape)
    print('Extra test shape:',extra_test.shape)
  X_train = features[X_train]; X_test = features[X_test]
  if EXTRA_INPUTS is not None:
    X_train = np.concatenate((X_train, extra_train), axis=-1)
    X_test = np.concatenate((X_test, extra_test), axis=-1)
    print('>>> Extra inputs concatenated to features')
    print('X_train shape:',X_train.shape)
    print('X_test shape:',X_test.shape)
  del features; del labels # memory purposes
  if BINARY_MASK:
    Y_train = nets.convert_to_binary_mask(Y_train)
    Y_test = nets.convert_to_binary_mask(Y_test)
    print('>>> Converted to binary mask!')
    print('Y_train shape:',Y_train.shape)
    print('Y_test shape:',Y_test.shape)
    print('Y_train unique:',np.unique(Y_train))
    print('Y_test unique:',np.unique(Y_test))
    print(f'Y_train summary: {plotter.summary(Y_train)}')
    print(f'Y_test summary: {plotter.summary(Y_test)}')
  # change labels dtype to int8
  Y_train = Y_train.astype(np.int8)
  Y_test = Y_test.astype(np.int8)
  print('>>> Labels dtype changed to int8')
  print(f'>>> Split into training ({(1-test_size)*100}%) and validation ({test_size*100}%) sets')
  print('X_train shape: ',X_train.shape); print('Y_train shape: ',Y_train.shape)
  print('X_test shape: ',X_test.shape); print('Y_test shape: ',Y_test.shape)
  print('X_train dtype:',X_train.dtype); print('Y_train dtype:',Y_train.dtype)
  print('X_test dtype:',X_test.dtype); print('Y_test dtype:',Y_test.dtype)
  if (LOSS != 'SCCE' and LOSS != 'DISCCE'):
    if not BINARY_MASK:
      print('>>> Converting to one-hot encoding')
      Y_train = nets.to_categorical(Y_train, num_classes=N_CLASSES)
      Y_test  = nets.to_categorical(Y_test,  num_classes=N_CLASSES)
      print('>>> One-hot encoding complete')
      print('Y_train shape: ',Y_train.shape)
      print('Y_test shape: ',Y_test.shape)
  #===============================================================
  # Save training and test arrays for later preds:
  # only need SIM, L, GRID to define filenames
  # filenames: TNG: TNG_L{L}_GRID{GRID}_X_test.npy
  # BOL: BOL_L{L}_GRID{GRID}_X_test.npy
  # for Y_test though, the only things that matter is SIM, GRID,
  # and if they're one-hot encoded or not. if LOSS = SCCE,
  # add a flag that indicates the labels are integers.
  #===============================================================
  if os.path.exists(FILE_X_TEST) and os.path.exists(FILE_Y_TEST):
    print(f'Files {FILE_X_TEST} and {FILE_Y_TEST} already exist.')
  elif os.path.exists(FILE_X_TEST) and not os.path.exists(FILE_Y_TEST):
    np.save(FILE_Y_TEST,Y_test,allow_pickle=True)
    print(f'File {FILE_X_TEST} already exists.')
    print(f'>>> Saved Y_test to {FILE_Y_TEST}')
  elif not os.path.exists(FILE_X_TEST) and os.path.exists(FILE_Y_TEST):
    np.save(FILE_X_TEST,X_test,allow_pickle=True)
    print(f'File {FILE_Y_TEST} already exists.')
    print(f'>>> Saved X_test to {FILE_X_TEST}')
  elif not os.path.exists(FILE_X_TEST) and not os.path.exists(FILE_Y_TEST):
    np.save(FILE_X_TEST,X_test,allow_pickle=True)
    np.save(FILE_Y_TEST,Y_test,allow_pickle=True)
    print(f'>>> Saved X_test to {FILE_X_TEST}')
    print(f'>>> Saved Y_test to {FILE_Y_TEST}')
  #===============================================================
  # Make tf.data.Dataset
  #===============================================================
  if BOUNDARY_MASK is not None:
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train, boundary_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test, boundary_test))
  else:
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
else:
  print('>>> Loading train, val data into tf.data.Dataset from memmapped .npy files')
  print('>>> Loading X_train, Y_train, X_test, Y_test from .npy files')
  print('>>> NOTE: X_train, Y_train, X_test, Y_test will not be loaded into memory!')
  print('>>> X_train:',FILE_X_TRAIN); print('>>> Y_train:',FILE_Y_TRAIN)
  print('>>> X_test:',FILE_X_TEST); print('>>> Y_test:',FILE_Y_TEST)
  n_samples_train = np.load(FILE_X_TRAIN,mmap_mode='r').shape[0]
  n_samples_test = np.load(FILE_X_TEST,mmap_mode='r').shape[0]
  last_dim = 1 if LOSS == 'SCCE' or LOSS == 'DISCCE' else N_CLASSES
  if BINARY_MASK:
    last_dim = 1
  train_dataset = tf.data.Dataset.from_generator(
    lambda: nets.data_gen_mmap(FILE_X_TRAIN,FILE_Y_TRAIN),
    output_signature=(
      tf.TensorSpec(shape=(SUBGRID,SUBGRID,SUBGRID,1),dtype=tf.float32),
      tf.TensorSpec(shape=(SUBGRID,SUBGRID,SUBGRID,last_dim),dtype=tf.int8)
    )
  )
  test_dataset = tf.data.Dataset.from_generator(
    lambda: nets.data_gen_mmap(FILE_X_TEST,FILE_Y_TEST),
    output_signature=(
      tf.TensorSpec(shape=(SUBGRID,SUBGRID,SUBGRID,1),dtype=tf.float32),
      tf.TensorSpec(shape=(SUBGRID,SUBGRID,SUBGRID,last_dim),dtype=tf.int8)
    )
  )
# 5/28 try caching to see if it speeds up training
# NOTE: results in OOM on colab
#train_dataset = train_dataset.cache()
#test_dataset = test_dataset.cache()
# shuffle and batch the datasets
print('>>> Shuffling and batching datasets')
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)
test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
# prefetch the datasets
print('>>> Prefetching datasets')
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
# manually set cardinality of datasets
if LOAD_INTO_MEM:
  # do i even need to do this? won't cardinality be set?
  pass
else:
  cardinality_train = n_samples_train // batch_size
  cardinality_test = n_samples_test // batch_size
  train_dataset = train_dataset.apply(tf.data.experimental.assert_cardinality(cardinality_train))
  test_dataset = test_dataset.apply(tf.data.experimental.assert_cardinality(cardinality_test))
  print('>>> Cardinality of train dataset:',cardinality_train)
  print('>>> Cardinality of test dataset:',cardinality_test)
#===============================================================
# Set model hyperparameters
#===============================================================
MODEL_NAME = nets.create_model_name(
  SIM,DEPTH,FILTERS,GRID,th,sig,L,
  UNIFORM_FLAG,BATCHNORM,DROPOUT,LOSS,
  suffix=MODEL_NAME_SUFFIX if MODEL_NAME_SUFFIX else None
  )
print('>>> Model name:',MODEL_NAME)
DATE = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#===============================================================
# Save hyperparameters to txt file
#===============================================================
hp_dict = {}
hp_dict['DATE_CREATED'] = DATE
hp_dict['MODEL_NAME'] = MODEL_NAME
hp_dict['notes'] = f'trained on multi-class mask, threshold={th}, sigma={sig}, L={L}, Nm={GRID}'
hp_dict['KERNEL'] = KERNEL
if SIM == 'TNG':
  hp_dict['Simulation trained on'] = 'TNG300-3-Dark'
elif SIM == 'BOL':
  hp_dict['Simulation trained on'] = 'Bolshoi'
hp_dict['N_CLASSES'] = N_CLASSES
hp_dict['DEPTH'] = DEPTH
hp_dict['FILTERS'] = FILTERS
hp_dict['LR'] = LR
hp_dict['LOSS'] = LOSS
if LOSS == 'FOCAL_CCE':
  hp_dict['focal_alpha'] = alpha
  hp_dict['focal_gamma'] = gamma
hp_dict['BATCHNORM'] = str(BATCHNORM)
hp_dict['DROPOUT'] = str(DROPOUT)
hp_dict['REGULARIZE_FLAG'] = str(REGULARIZE_FLAG)
hp_dict['BATCH_SIZE'] = batch_size
hp_dict['MAX_EPOCHS'] = epochs
hp_dict['PATIENCE'] = PATIENCE
hp_dict['LR_PATIENCE'] = LR_PATIENCE
hp_dict['FILE_DEN'] = FILE_DEN
hp_dict['FILE_MASK'] = FILE_MASK
hp_dict['FILE_X_TRAIN'] = FILE_X_TRAIN
hp_dict['FILE_Y_TRAIN'] = FILE_Y_TRAIN
hp_dict['FILE_X_TEST'] = FILE_X_TEST
hp_dict['FILE_Y_TEST'] = FILE_Y_TEST
hp_dict['GRID'] = GRID
hp_dict['SUBGRID'] = SUBGRID
hp_dict['OFF'] = OFF
hp_dict['UNIFORM_FLAG'] = UNIFORM_FLAG
hp_dict['ADD_RSD'] = ADD_RSD
hp_dict['PARTIAL_CONV'] = USE_PCONV
if BOUNDARY_MASK is not None:
  hp_dict['BOUNDARY_MASK'] = BOUNDARY_MASK
print('#############################################')
print('>>> Model Hyperparameters:')
for key in hp_dict.keys():
  print(key,hp_dict[str(key)])
print('#############################################')
#===============================================================
# Set loss function and metrics
#===============================================================
metrics = ['accuracy']
if LOSS == 'CCE':
  loss = nets.CategoricalCrossentropy()
elif LOSS == 'SCCE':
  loss = nets.SparseCategoricalCrossentropy()
elif LOSS == 'FOCAL_CCE':
  loss = [nets.categorical_focal_loss(alpha=alpha,gamma=gamma)]
  #loss = nets.CategoricalFocalCrossentropy(alpha=alpha,gamma=gamma) # only works on 2.16.1
elif LOSS == 'DISCCE':
  # implement dice loss averaged over all classes
  loss = [nets.SCCE_Dice_loss]
elif LOSS == 'BCE':
  loss = nets.BinaryCrossentropy()
  if not BINARY_MASK:
    sys.exit('ERROR: Binary crossentropy loss only compatible with binary mask')
elif LOSS == 'SCCE_Void_Penalty':
  loss_fn = [nets.SCCE_void_penalty]
elif LOSS == 'DICE_VOID':
  # implement dice loss with void class
  pass
# set one-hot flag:
ONE_HOT_FLAG = True # for compute metrics callback
if LOSS == 'SCCE' or LOSS == 'DISCCE':
  ONE_HOT_FLAG = False
if BINARY_MASK:
  ONE_HOT_FLAG = False
print('>>> One-hot flag:',ONE_HOT_FLAG)
# add more metrics here, may slow down training?
more_metrics = [nets.MCC_keras(int_labels=not ONE_HOT_FLAG),nets.balanced_accuracy_keras(int_labels=not ONE_HOT_FLAG),
                nets.void_F1_keras(int_labels=not ONE_HOT_FLAG),nets.F1_micro_keras(int_labels=not ONE_HOT_FLAG)]
if not LOW_MEM_FLAG:
  more_metrics += [nets.recall_micro_keras(int_labels=not ONE_HOT_FLAG),
                   nets.precision_micro_keras(int_labels=not ONE_HOT_FLAG),
                   nets.true_wall_pred_as_void_keras(int_labels=not ONE_HOT_FLAG)]
metrics += more_metrics
if BINARY_MASK:
  metrics = ['accuracy']
# print metrics:
print('>>> Metrics:')
for metric in metrics:
  print(str(metric))
#===============================================================
# Multiprocessing
#===============================================================
if BINARY_MASK:
  N_CLASSES = 1 # just for unet_3d() call
last_activation = 'softmax' if N_CLASSES > 1 else 'sigmoid'
print('>>> Last activation function:',last_activation)
input_shape = (None, None, None, 1) # default input shape for 3D U-Net
if EXTRA_INPUTS is not None:
  input_shape = (None, None, None, N_CHANNELS)
if MULTI_FLAG:
  strategy = tf.distribute.MirroredStrategy()
  print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
  with strategy.scope():
    # if model_name exists in FILE_OUT, load it
    # if not, create a new model
    if os.path.exists(FILE_OUT+MODEL_NAME) and LOAD_MODEL_FLAG:
      print('>>> Loaded model:',FILE_OUT+MODEL_NAME)
      model = nets.load_model(FILE_OUT+MODEL_NAME)
      model.set_weights(model.get_weights())
    else:
      if not USE_PCONV: 
        model = nets.unet_3d(input_shape,N_CLASSES,FILTERS,DEPTH,
                            last_activation=last_activation,
                            batch_normalization=BATCHNORM,
                            dropout_rate=DROPOUT,
                            model_name=MODEL_NAME,
                            REG_FLAG=REGULARIZE_FLAG)
      else:
        model = nets.unet_partial_conv_3d_with_survey_mask(
          input_shape, initial_filters=FILTERS, depth=DEPTH,
          last_activation=last_activation
        )
      model.compile(optimizer=nets.Adam(learning_rate=LR),
                                        loss=loss,
                                        metrics=metrics)
else:
  if os.path.exists(FILE_OUT+MODEL_NAME) and LOAD_MODEL_FLAG:
    print('>>> Loaded model:',FILE_OUT+MODEL_NAME)
    model = nets.load_model(FILE_OUT+MODEL_NAME)
    model.set_weights(model.get_weights())
  else:
    if not USE_PCONV:
      model = nets.unet_3d(input_shape,N_CLASSES,FILTERS,DEPTH,
                          last_activation=last_activation,
                          batch_normalization=BATCHNORM,
                          dropout_rate=DROPOUT,
                          model_name=MODEL_NAME,
                          REG_FLAG=REGULARIZE_FLAG)
    else:
      model = nets.unet_partial_conv_3d_with_survey_mask(
        input_shape, initial_filters=FILTERS, depth=DEPTH,
        last_activation=last_activation
      )
    model.compile(optimizer=nets.Adam(learning_rate=LR),
                                          loss=loss,
                                          metrics=metrics)
model.summary()
# get trainable parameters:
trainable_ps = nets.layer_utils.count_params(model.trainable_weights)
nontrainable_ps = nets.layer_utils.count_params(model.non_trainable_weights)
hp_dict['trainable_params'] = trainable_ps
hp_dict['nontrainable_params'] = nontrainable_ps
hp_dict['total_params'] = trainable_ps + nontrainable_ps
# save hyperparameters to file:
FILE_HPS = FILE_OUT+MODEL_NAME+'_hps.txt'
print('>>> Saving hyperparameters to:',FILE_HPS)
nets.save_dict_to_text(hp_dict,FILE_HPS)
#===============================================================
# Train
# python3 -m tensorboard.main --logdir=./logs
# (^^^^ cmd for tensorboard, must be on sciserver conda env)
#===============================================================
print('>>> Training')
# set up callbacks
#metrics = nets.ComputeMetrics((X_test,Y_test), N_epochs = N_epochs_metric, avg='micro', one_hot=ONE_HOT_FLAG)
model_chkpt = nets.ModelCheckpoint(FILE_OUT + MODEL_NAME + '.keras', monitor='val_loss',
                                   save_best_only=True,verbose=2)
log_dir = ROOT_DIR + 'logs/fit/' + MODEL_NAME + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M") 
tb_call = nets.TensorBoard(log_dir=log_dir) # do we even need this if we CSV log?
csv_logger = nets.CSVLogger(FILE_OUT+MODEL_NAME+'_' + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '_train_log.csv')
reduce_lr = nets.ReduceLROnPlateau(monitor='val_loss',factor=0.25,patience=LR_PATIENCE, 
                                   verbose=1,min_lr=1e-7)
early_stop = nets.EarlyStopping(monitor='val_loss',patience=PATIENCE,restore_best_weights=True)
if LOW_MEM_FLAG:
  # dont calc metrics, too memory intensive
  callbacks = [model_chkpt,reduce_lr,early_stop,csv_logger]
else:
  #callbacks = [metrics,model_chkpt,reduce_lr,early_stop,csv_logger]
  callbacks = [model_chkpt,reduce_lr,early_stop,csv_logger]
if TENSORBOARD_FLAG:
  callbacks.append(tb_call)
#===============================================================
# Train model
#===============================================================
history = model.fit(
  train_dataset, epochs=epochs, validation_data=test_dataset, verbose=2,
  callbacks=callbacks
)
#===============================================================
# Check if figs directory exists, if not, create it:
#===============================================================
print('>>> Plotting training metrics')
FIG_DIR = FILE_FIG + MODEL_NAME + '/'
if not os.path.exists(FIG_DIR):
  os.makedirs(FIG_DIR)
  print('>>> Created directory for figures:',FIG_DIR)
FILE_METRICS = FIG_DIR + MODEL_NAME + '_metrics.png'
plotter.plot_training_metrics_all(history,FILE_METRICS,savefig=True)
#===============================================================
# set up score_dict. 
# VAL_FLAG is True if scores are based on val set
# ORTHO_FLAG is True if scores are based on orthogonal rotated delta/mask
# for 45 deg rotated cubes, ORTHO_FLAG = False
#===============================================================
VAL_FLAG = True 
ORTHO_FLAG = True 
scores = {}
scores['SIM'] = SIM; scores['DEPTH'] = DEPTH; scores['FILTERS'] = FILTERS
scores['L_TRAIN'] = L; scores['L_PRED'] = L
scores['UNIFORM_FLAG'] = UNIFORM_FLAG; scores['BATCHNORM'] = BATCHNORM
scores['DROPOUT'] = DROPOUT; scores['LOSS'] = LOSS
scores['GRID'] = GRID; scores['DATE'] = DATE; scores['MODEL_NAME'] = MODEL_NAME
scores['VAL_FLAG'] = VAL_FLAG
scores['ORTHO_FLAG'] = ORTHO_FLAG
epochs = len(history.epoch)
scores['EPOCHS'] = epochs
scores['BATCHSIZE'] = batch_size
scores['LR'] = LR
scores['REG_FLAG'] = REGULARIZE_FLAG
scores['TRAINABLE_PARAMS'] = trainable_ps
scores['NONTRAINABLE_PARAMS'] = nontrainable_ps
scores['TOTAL_PARAMS'] = trainable_ps + nontrainable_ps
scores['TRAIN_LOSS'] = history.history['loss'][-1]
scores['VAL_LOSS'] = history.history['val_loss'][-1]
scores['TRAIN_ACC'] = history.history['accuracy'][-1]
scores['VAL_ACC'] = history.history['val_accuracy'][-1]
if LOSS == 'FOCAL_CCE':
  scores['FOCAL_ALPHA'] = alpha
  scores['FOCAL_GAMMA'] = gamma
if EXTRA_INPUTS is not None:
  scores['EXTRA_INPUTS'] = EXTRA_INPUTS
#===============================================================
# Predict, record metrics, and plot metrics on TEST DATA
#===============================================================
if LOAD_INTO_MEM:
  print('>>> Predicting on test data')
  Y_pred = nets.run_predict_model(model,X_test,batch_size,output_argmax=False)
  # since output argmax = False, Y_pred shape = [N_samples,SUBGRID,SUBGRID,SUBGRID,N_CLASSES]
else:
  print('>>> Predicting on test data, loading in batches')
  Y_pred_list = []; Y_test_list = []
  for X_batch, Y_batch in test_dataset:
    Y_pred_batch = model.predict(X_batch,verbose=0)
    Y_pred_list.append(Y_pred_batch)
    Y_test_list.append(Y_batch.numpy()) # NOTE this may OOM???
  Y_pred = np.concatenate(Y_pred_list,axis=0)
  Y_test = np.concatenate(Y_test_list,axis=0)
# save Y_pred as is:
np.save(FILE_OUT+MODEL_NAME+'_Y_pred.npy',Y_pred,allow_pickle=True)
# adjust Y_test shape to be [N_samples,SUBGRID,SUBGRID,SUBGRID,1]:
if (LOSS != 'SCCE' and LOSS != 'DISCCE'):
  if not BINARY_MASK:
    # undo one-hot encoding for input into save_scores_from_fvol
    Y_test = np.argmax(Y_test,axis=-1)
    Y_test = np.expand_dims(Y_test,axis=-1)
#print('Y_pred summary:',np.unique(Y_pred,return_counts=True))
#print('Y_test summary:',np.unique(Y_test,return_counts=True))
print('Y_pred shape:',Y_pred.shape)
print('Y_test shape:',Y_test.shape)
# save scores
print('>>> Calculating scores on validation data')
if BINARY_MASK:
  N_CLASSES = 2 # janky fix for save_scores_from_fvol
nets.save_scores_from_fvol(Y_test,Y_pred,
                           FILE_OUT+MODEL_NAME,FIG_DIR,
                           scores,
                           N_CLASSES=N_CLASSES,
                           VAL_FLAG=VAL_FLAG)
# save score_dict by appending to the end of the csv.
# csv will be at ROOT_DIR/model_scores.csv
print(f'>>> Saving all scores to {ROOT_DIR}/model_scores.csv')
nets.save_scores_to_csv(scores,ROOT_DIR+'model_scores.csv')
print(f'>>> Saving score summary to {ROOT_DIR}/model_scores_summary.csv')
nets.save_scores_to_csv_small(scores,ROOT_DIR+'model_scores_summary.csv')
#========================================================================
# Predict and plot and record metrics on TRAINING DATA
# with TRAIN_SCORE = False, all this does is predict on the entire 
# data cube and save slices of the predicted mask 
# for slice plotting:
#========================================================================
print('>>> Predicting on training data and plotting slices')
if SIM == 'TNG':
  nets.save_scores_from_model(FILE_DEN, FILE_MASK, FILE_OUT+MODEL_NAME, FIG_DIR, FILE_PRED,
                              GRID=GRID,SUBGRID=SUBGRID,OFF=OFF,TRAIN_SCORE=False,
                              BINARY=BINARY_MASK,EXTRA_INPUTS=EXTRA_INPUTS)
elif SIM == 'BOL':
  nets.save_scores_from_model(FILE_DEN, FILE_MASK, FILE_OUT+MODEL_NAME, FIG_DIR, FILE_PRED,
                              GRID=GRID,SUBGRID=SUBGRID,OFF=OFF,BOXSIZE=256,BOLSHOI_FLAG=True,
                              TRAIN_SCORE=False,BINARY=BINARY_MASK,EXTRA_INPUTS=EXTRA_INPUTS)
print('>>> Finished predicting on training data')
#===============================================================
print('Finished training!')
print('Model name:',MODEL_NAME)
print('Interparticle spacing model trained on:',L)
print(f'Model parameters: Depth={DEPTH}, Filters={FILTERS}, Uniform={UNIFORM_FLAG}, BatchNorm={BATCHNORM}, Dropout={DROPOUT}')
print(f'Loss function: {LOSS}')
print('Date created:',DATE)
#print('Total trainable parameters:',trainable_ps)
#print('Total nontrainable parameters:',nontrainable_ps)
#print('Total parameters:',trainable_ps+nontrainable_ps)
print('>>> Finished training!!!')
#===============================================================