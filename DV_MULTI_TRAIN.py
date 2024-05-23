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
print('TensorFlow version: ', tf.__version__)
#print('Keras version: ', tf.keras.__version__)
print('CUDA?: ',tf.test.is_built_with_cuda())
nets.K.set_image_data_format('channels_last')
# only use with Nvidia GPUs with compute capability >= 7.0!
#from tensorflow.keras import mixed_precision  # type: ignore
#mixed_precision.set_global_policy('mixed_float16')
#===============================================================
# Set training parameters:
#===============================================================
# set to True if you want to use less memory, but no metrics and 
# less subcubes loaded into memory at once.
LOW_MEM_FLAG = True 
epochs = 500; print('epochs: ',epochs)
patience = 50; print('patience: ',patience)
lr_patience = 20; print('learning rate patience: ',lr_patience)
# batch_size = 8; print('batch_size: ',batch_size) # set in arg parsing
N_epochs_metric = 10
print(f'classification metrics calculated every {N_epochs_metric} epochs')
KERNEL = (3,3,3)
LR = 3e-3 # increased to 3e-3 since we have LRreduceonplateau anyway
#===============================================================
# Set random seed
#===============================================================
seed = 12
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
  L: Interparticle separation in Mpc/h. For TNG full DM use '0.33', for BOL full DM use '0.122'. Other valid values are '3', '5', '7', '10'.
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
  --LOAD_MODEL_FLAG: If set, load model from FILE_OUT if it exists. Default is False.
  --LOAD_INTO_MEM: If set, load training and test data into memory. 
  If not set, load data from X_train, Y_train, X_test, Y_test .npy files into a tf.data.Dataset object 
  that will load the data in batches instead of all at once. Default is False.
  --BATCH_SIZE: Batch size. Default is 4.
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
req_group.add_argument('LOSS', type=str, default='CCE', help='Loss function to use: CCE, SCCE, FOCAL_CCE.')
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
opt_group.add_argument('--LOAD_MODEL_FLAG', action='store_true', help='If set, load model from FILE_OUT if it exists.')
opt_group.add_argument('--LOAD_INTO_MEM', action='store_true', help='If set, load all training and test data into memory. Default is False, aka to load from train, test .npy files into a tf.data.Dataset object.')
opt_group.add_argument('--BATCH_SIZE', type=int, default=8, help='Batch size. Default is 4.')
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
LOAD_MODEL_FLAG = args.LOAD_MODEL_FLAG
LOAD_INTO_MEM = args.LOAD_INTO_MEM
batch_size = args.BATCH_SIZE
print('#############################################')
print('>>> Running DV_MULTI_TRAIN.py')
print('>>> Root directory:',ROOT_DIR)
print('>>> Parameters:')
print('Simulation =', SIM); 
print('L =',L); 
print('DEPTH =',DEPTH); print('FILTERS =',FILTERS)
print('UNIFORM_FLAG =',UNIFORM_FLAG)
print('BATCHNORM =',BATCHNORM)
print('DROPOUT =',DROPOUT)
print('LOSS =',LOSS)
if LOSS == 'FOCAL_CCE':
  print('FOCAL_ALPHA =',alpha)
  print('FOCAL_GAMMA =',gamma)
print('MULTI_FLAG =',MULTI_FLAG)
print('GRID =',GRID)
print('BATCH_SIZE =',batch_size)
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
# set up .npy filepaths for saving/loading data
X_PREFIX = f'{SIM}_L{L}_Nm={GRID}'
Y_PREFIX = f'{SIM}_Nm={GRID}'
if LOSS == 'SCCE':
  Y_PREFIX += '_int'
### TNG ### 
if SIM == 'TNG':
  BoxSize = 205.0 # Mpc/h
  #GRID = 512 
  SUBGRID = 128
  OFF = 64
  ### TNG Density field:
  # FULL (L=0.33 Mpc/h)
  FILE_DEN_FULL = path_to_TNG + f'DM_DEN_snap99_Nm={GRID}.fvol'
  if UNIFORM_FLAG == True:
    FILE_DEN_SUBS = path_to_TNG + f'subs1_mass_Nm{GRID}_L{L}_d_None_smooth_uniform.fvol'
  if UNIFORM_FLAG == False:
    FILE_DEN_SUBS = path_to_TNG + f'subs1_mass_Nm{GRID}_L{L}_d_None_smooth.fvol'
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
    features, labels = nets.load_dataset_all(FILE_DEN,FILE_MASK,SUBGRID)
  else:
    features, labels = nets.load_dataset_all_overlap(FILE_DEN,FILE_MASK,SUBGRID,OFF)
  print('>>> Data loaded!'); print('Features shape:',features.shape)
  print('Labels shape:',labels.shape)
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
  X_train = features[X_train]; X_test = features[X_test]
  del features; del labels # memory purposes
  print(f'>>> Split into training ({(1-test_size)*100}%) and validation ({test_size*100}%) sets')
  print('X_train shape: ',X_train.shape); print('Y_train shape: ',Y_train.shape)
  print('X_test shape: ',X_test.shape); print('Y_test shape: ',Y_test.shape)
  if LOSS != 'SCCE':
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
  train_dataset = tf.data.Dataset.from_tensor_slices((X_train,Y_train))
  test_dataset = tf.data.Dataset.from_tensor_slices((X_test,Y_test))
else:
  print('>>> Loading train, val data into tf.data.Dataset from memmapped .npy files')
  print('>>> Loading X_train, Y_train, X_test, Y_test from .npy files')
  print('>>> NOTE: X_train, Y_train, X_test, Y_test will not be loaded into memory!')
  print('>>> X_train:',FILE_X_TRAIN); print('>>> Y_train:',FILE_Y_TRAIN)
  print('>>> X_test:',FILE_X_TEST); print('>>> Y_test:',FILE_Y_TEST)
  last_dim = 1 if LOSS == 'SCCE' else N_CLASSES
  train_dataset = tf.data.Dataset.from_generator(
    lambda: nets.data_gen_mmap(FILE_X_TRAIN,FILE_Y_TRAIN),
    output_signature=(
      tf.TensorSpec(shape=(SUBGRID,SUBGRID,SUBGRID,1),dtype=tf.float32),
      tf.TensorSpec(shape=(SUBGRID,SUBGRID,SUBGRID,last_dim),dtype=tf.float32)
    )
  )
  test_dataset = tf.data.Dataset.from_generator(
    lambda: nets.data_gen_mmap(FILE_X_TEST,FILE_Y_TEST),
    output_signature=(
      tf.TensorSpec(shape=(SUBGRID,SUBGRID,SUBGRID,1),dtype=tf.float32),
      tf.TensorSpec(shape=(SUBGRID,SUBGRID,SUBGRID,last_dim),dtype=tf.float32)
    )
  )
# shuffle and batch the datasets
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)
# prefetch the datasets
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
#===============================================================
# Set model hyperparameters
#===============================================================
if SIM == 'TNG':
  MODEL_NAME = f'TNG_D{DEPTH}-F{FILTERS}-Nm{GRID}-th{th}-sig{sig}-base_L{L}'
elif SIM == 'BOL':
  MODEL_NAME = f'Bolshoi_D{DEPTH}-F{FILTERS}-Nm{GRID}-th{th}-sig{sig}-base_L{L}'
if UNIFORM_FLAG:
  MODEL_NAME += '_uniform'
if BATCHNORM:
  MODEL_NAME += '_BN'
if DROPOUT != 0.0:
  MODEL_NAME += f'_DROP{DROPOUT}'
if LOSS == 'CCE':
  pass
elif LOSS == 'FOCAL_CCE':
  MODEL_NAME += '_FOCAL'
elif LOSS == 'SCCE':
  MODEL_NAME += '_SCCE'
  print('Loss function SCCE requires integer labels, NOT one-hots')
### NOTE: add support for more loss functions here NOTE ###
DATE = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#===============================================================
# Save hyperparameters to txt file
#===============================================================
hp_dict = {}
hp_dict['notes'] = f'trained on multi-class mask, threshold={th}, sigma={sig}, L={L}, Nm={GRID}'
if SIM == 'TNG':
  hp_dict['Simulation trained on:'] = 'TNG300-3-Dark'
elif SIM == 'BOL':
  hp_dict['Simulation trained on:'] = 'Bolshoi'
hp_dict['N_CLASSES'] = N_CLASSES
hp_dict['MODEL_NAME'] = MODEL_NAME
hp_dict['FILTERS'] = FILTERS
hp_dict['KERNEL'] = KERNEL
hp_dict['LR'] = LR
hp_dict['DEPTH'] = DEPTH
hp_dict['LOSS'] = LOSS
if LOSS == 'FOCAL_CCE':
  hp_dict['focal_alpha'] = alpha
  hp_dict['focal_gamma'] = gamma
hp_dict['BATCHNORM'] = str(BATCHNORM)
hp_dict['DROPOUT'] = str(DROPOUT)
hp_dict['DATE_CREATED'] = DATE
hp_dict['FILE_DEN'] = FILE_DEN
hp_dict['FILE_MASK'] = FILE_MASK
for key in hp_dict.keys():
  print(key,hp_dict[str(key)])
#===============================================================
# Set loss function and metrics
#===============================================================
metrics = ['accuracy']
if LOSS == 'CCE':
  loss = nets.CategoricalCrossentropy()
elif LOSS == 'SCCE':
  loss = nets.SparseCategoricalCrossentropy()
  # replace 'categorical_accuracy' with 'sparse_categorical_accuracy'
  metrics = ['accuracy']
elif LOSS == 'FOCAL_CCE':
  loss = [nets.categorical_focal_loss(alpha=alpha,gamma=gamma)]
  #loss = nets.CategoricalFocalCrossentropy(alpha=alpha,gamma=gamma) # only works on 2.16.1
elif LOSS == 'DICE_AVG':
  # implement dice loss averaged over all classes
  pass
elif LOSS == 'DICE_VOID':
  # implement dice loss with void class
  pass
# add more metrics here, may slow down training?
if not LOW_MEM_FLAG:
  #metrics += ['f1_score','precision','recall'] # not on tf 2.10.1
  pass
#===============================================================
# Multiprocessing
#===============================================================
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
      model = nets.unet_3d((None,None,None,1),N_CLASSES,FILTERS,DEPTH,
                          batch_normalization=BATCHNORM,
                          dropout_rate=DROPOUT,
                          model_name=MODEL_NAME)
      model.compile(optimizer=nets.Adam(learning_rate=LR),
                                        loss=loss,
                                        metrics=metrics)
else:
  if os.path.exists(FILE_OUT+MODEL_NAME) and LOAD_MODEL_FLAG:
    print('>>> Loaded model:',FILE_OUT+MODEL_NAME)
    model = nets.load_model(FILE_OUT+MODEL_NAME)
    model.set_weights(model.get_weights())
  else:
    model = nets.unet_3d((None,None,None,1),N_CLASSES,FILTERS,DEPTH,
                          batch_normalization=BATCHNORM,
                          dropout_rate=DROPOUT,
                          model_name=MODEL_NAME)
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
nets.save_dict_to_text(hp_dict,FILE_HPS)
#===============================================================
# Train
# python3 -m tensorboard.main --logdir=./logs
# (^^^^ cmd for tensorboard, must be on sciserver conda env)
#===============================================================
print('>>> Training')
# set up callbacks
ONE_HOT_FLAG = True # for compute metrics callback
if LOSS == 'SCCE':
  ONE_HOT_FLAG = False
#metrics = nets.ComputeMetrics((X_test,Y_test), N_epochs = N_epochs_metric, avg='micro', one_hot=ONE_HOT_FLAG)
model_chkpt = nets.ModelCheckpoint(FILE_OUT + MODEL_NAME + '.keras', monitor='val_loss',
                                   save_best_only=True,verbose=2)
#log_dir = "logs/fit/" + MODEL_NAME + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M") 
#tb_call = nets.TensorBoard(log_dir=log_dir) # do we even need this if we CSV log?
csv_logger = nets.CSVLogger(FILE_OUT+MODEL_NAME+'_' + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '_train_log.csv')
reduce_lr = nets.ReduceLROnPlateau(monitor='val_loss',factor=0.25,patience=lr_patience, 
                                   verbose=1,min_lr=1e-6)
early_stop = nets.EarlyStopping(monitor='val_loss',patience=patience,restore_best_weights=True)
if LOW_MEM_FLAG:
  # dont calc metrics, too memory intensive
  callbacks = [model_chkpt,reduce_lr,early_stop,csv_logger]
else:
  #callbacks = [metrics,model_chkpt,reduce_lr,early_stop,csv_logger]
  callbacks = [model_chkpt,reduce_lr,early_stop,csv_logger]
#history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
#                    validation_data=(X_test,Y_test), verbose = 2, shuffle = True,
#                    callbacks = callbacks)
history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=2,
                    callbacks=callbacks)
#===============================================================
# Check if figs directory exists, if not, create it:
#===============================================================
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
scores['TRAINABLE_PARAMS'] = trainable_ps
scores['NONTRAINABLE_PARAMS'] = nontrainable_ps
scores['TOTAL_PARAMS'] = trainable_ps + nontrainable_ps
scores['TRAIN_LOSS'] = history.history['loss'][-1]
scores['VAL_LOSS'] = history.history['val_loss'][-1]
scores['TRAIN_ACC'] = history.history['accuracy'][-1]
scores['VAL_ACC'] = history.history['val_accuracy'][-1]
if LOSS == 'SCCE':
  scores['TRAIN_CAT_ACC'] = history.history['sparse_categorical_accuracy'][-1]
  scores['VAL_CAT_ACC'] = history.history['val_sparse_categorical_accuracy'][-1]
else:
  scores['TRAIN_CAT_ACC'] = history.history['categorical_accuracy'][-1]
  scores['VAL_CAT_ACC'] = history.history['val_categorical_accuracy'][-1]
if LOSS == 'FOCAL_CCE':
  scores['FOCAL_ALPHA'] = alpha
  scores['FOCAL_GAMMA'] = gamma
#===============================================================
# Predict, record metrics, and plot metrics on TEST DATA
#===============================================================
if LOAD_INTO_MEM:
  Y_pred = nets.run_predict_model(model,X_test,batch_size,output_argmax=False)
  # since output argmax = False, Y_pred shape = [N_samples,SUBGRID,SUBGRID,SUBGRID,N_CLASSES]
else:
  Y_pred_list = []; Y_test_list = []
  for X_batch, Y_batch in test_dataset:
    Y_pred_batch = model.predict(X_batch,verbose=0)
    Y_pred_list.append(Y_pred_batch)
    Y_test_list.append(Y_batch.numpy()) # NOTE this may OOM???
  Y_pred = np.concatenate(Y_pred_list,axis=0)
  Y_test = np.concatenate(Y_test_list,axis=0)
# adjust Y_test shape to be [N_samples,SUBGRID,SUBGRID,SUBGRID,1]:
if LOSS != 'SCCE':
  # undo one-hot encoding for input into save_scores_from_fvol
  Y_test = np.argmax(Y_test,axis=-1)
  Y_test = np.expand_dims(Y_test,axis=-1)
nets.save_scores_from_fvol(Y_test,Y_pred,
                           FILE_OUT+MODEL_NAME,FIG_DIR,
                           scores,
                           VAL_FLAG=VAL_FLAG)
# save score_dict by appending to the end of the csv.
# csv will be at ROOT_DIR/model_scores.csv
nets.save_scores_to_csv(scores,ROOT_DIR+'model_scores.csv')
#========================================================================
# Predict and plot and record metrics on TRAINING DATA
# with TRAIN_SCORE = False, all this does is predict on the entire 
# data cube and save slices of the predicted mask 
# for slice plotting:
#========================================================================
if SIM == 'TNG':
  nets.save_scores_from_model(FILE_DEN, FILE_MASK, FILE_OUT+MODEL_NAME, FIG_DIR, FILE_PRED,
                              GRID=GRID,SUBGRID=SUBGRID,OFF=OFF,TRAIN_SCORE=False)
elif SIM == 'BOL':
  nets.save_scores_from_model(FILE_DEN, FILE_MASK, FILE_OUT+MODEL_NAME, FIG_DIR, FILE_PRED,
                              GRID=GRID,SUBGRID=SUBGRID,OFF=OFF,BOXSIZE=256,BOLSHOI_FLAG=True,
                              TRAIN_SCORE=False)
print('>>> Finished predicting on training data')
#===============================================================
print('Finished training!')
print('Model name:',MODEL_NAME)
print('Interparticle spacing model trained on:',L)
print(f'Model parameters: Depth={DEPTH}, Filters={FILTERS}, Uniform={UNIFORM_FLAG}, BatchNorm={BATCHNORM}, Dropout={DROPOUT}')
print(f'Loss function: {LOSS}')
print('Date created:',DATE)
print('Total trainable parameters:',trainable_ps)
print('Total nontrainable parameters:',nontrainable_ps)
print('Total parameters:',trainable_ps+nontrainable_ps)
print('>>> Finished training!!!')
#===============================================================