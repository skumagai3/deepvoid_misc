#!/usr/bin/env python3
'''
5/1/24: Making an updated version of dv-transfer-nonbinary.
NOTE that this is with the updated layer names.
'''
print('>>> Running DV_MULTI_TRANSFER.py')
import os
import ast
import sys
import datetime
import argparse
import numpy as np
import tensorflow as tf
import NETS_LITE as nets
import absl.logging
import plotter
absl.logging.set_verbosity(absl.logging.ERROR)
print('TensorFlow version: ', tf.__version__)
nets.K.set_image_data_format('channels_last')
# only use with Nvidia GPUs with compute capability >= 7.0!
#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')
#===============================================================
# Set training parameters:
#===============================================================
#patience = 25; print('patience: ',patience)
#lr_patience = 10; print('learning rate patience: ',lr_patience)
#N_epochs_metric = 10
#print(f'classification metrics calculated every {N_epochs_metric} epochs')
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
Usage: python3 DV_MULTI_TRANSFER.py <ROOT_DIR> <MODEL_NAME> <FN_DEN> <TL_TYPE> [--MULTI_FLAG] [--LOW_MEM_FLAG]

Arguments:
  ROOT_DIR: Root directory where data, models, figures, etc. are stored.
  MODEL_NAME: Name of the model to be loaded.
  FN_DEN: Filename of the density field to be loaded.
  TL_TYPE: Type of transfer learning to be done. Possible values: 'ENC', 'LL'.

Optional Flags:
  --MULTI_FLAG: If set, use multiprocessing. Default is False.
  --LOW_MEM_FLAG: If set, will load less training data and report fewer metrics. Default is True.
  --LOAD_INTO_MEM: If set, will load entire dataset into memory. Default is False.
  --TENSORBOARD_FLAG: If set, will use TensorBoard. Default is False.
  --EPOCHS: Number of epochs for training. Default is 500.
  --BATCH_SIZE: Batch size for training. Default is 8.
  --LEARNING_RATE: Initial learning rate. Default is 0.001.
  --LEARNING_RATE_PATIENCE: Number of epochs to wait before reducing learning rate. Default is 10.
  --PATIENCE: Number of epochs to wait before early stopping. Default is 25.

Notes:
MODEL_NAME (SIM, base_L will be pulled from that)
TL_TYPEs:
- ENC: freeze entire encoding side (and bottleneck)
- LL: freeze entire model except last conv block and output
- ENC_EO: freeze every other encoding conv block
- ENC_D1: freeze encoding side down to the second level
- ENC_D2: freeze encoding side down to the third level

Double transfer learning:
Transfer learning a model that has already been transfer learned
will append _TL_TYPE_{TL_TYPE}_tran_L={tran_L} to the MODEL_NAME
'''
parser = argparse.ArgumentParser(
    prog='DV_MULTI_TRANSFER.py',
    description='Transfer learning DV models to higher interparticle separations')
req_group = parser.add_argument_group('required arguments')
req_group.add_argument('ROOT_DIR',type=str, help='Root directory where data, models, figs, etc. are stored')
req_group.add_argument('MODEL_NAME',type=str, help='Name of the model to be loaded')
req_group.add_argument('FN_DEN',type=str, help='Filename of the density field to be loaded')
req_group.add_argument('TL_TYPE',type=str, help='Type of transfer learning to be done. Possible values: ENC, LL, ENC_EO')
opt_group = parser.add_argument_group('optional arguments')
opt_group.add_argument('--MULTI_FLAG',action='store_true',help='If set, use multiprocessing.')
opt_group.add_argument('--LOW_MEM_FLAG', action='store_false', help='If not set, will load less training data and report less metrics.')
opt_group.add_argument('--LOAD_INTO_MEM',action='store_true',help='If set, will load entire dataset into memory.')
opt_group.add_argument('--TENSORBOARD_FLAG',action='store_true',help='If set, will use TensorBoard.')
opt_group.add_argument('--EPOCHS',type=int, default=500, help='Number of epochs for training. Default is 500.')
opt_group.add_argument('--BATCH_SIZE',type=int, default=8, help='Batch size for training. Default is 8.')
opt_group.add_argument('--LEARNING_RATE',type=float, default=3e-3, help='Learning rate for training. Default is 3e-3.')
opt_group.add_argument('--LEARNING_RATE_PATIENCE',type=int, default=10, help='Patience for learning rate reduction. Default is 10.')
opt_group.add_argument('--PATIENCE',type=int, default=25, help='Patience for early stopping. Default is 25.')
opt_group.add_argument('--BINARY_FLAG',action='store_true',help='If set, will use binary mask for training.')
opt_group.add_argument('--EXTRA_INPUTS', type=str, default=None, help='If set, use extra inputs for the model. Should be a filename of a .fvol file.')
opt_group.add_argument('--UNIFORM_FLAG', type=int, default=1, help='If set to 1, will use uniform mass subsampling.')
args = parser.parse_args()
ROOT_DIR = args.ROOT_DIR
MODEL_NAME = args.MODEL_NAME
FN_DEN = args.FN_DEN
TL_TYPE = args.TL_TYPE
MULTI_FLAG = args.MULTI_FLAG
LOW_MEM_FLAG = args.LOW_MEM_FLAG
LOAD_INTO_MEM = args.LOAD_INTO_MEM
TENSORBOARD_FLAG = args.TENSORBOARD_FLAG
epochs = args.EPOCHS
batch_size = args.BATCH_SIZE
LR = args.LEARNING_RATE
lr_patience = args.LEARNING_RATE_PATIENCE
patience = args.PATIENCE
BINARY_MASK = args.BINARY_FLAG
UNIFORM_FLAG = args.UNIFORM_FLAG
EXTRA_INPUTS = args.EXTRA_INPUTS
if EXTRA_INPUTS is not None:
  print(f'Using extra input: {EXTRA_INPUTS}')
  N_CHANNELS = 2 # density + color fields NOTE can adjust later if we want to add more fields
print('#############################################')
#===============================================================
# hp dict is the old model, hp_dict_model is the new model
#===============================================================
hp_dict = nets.parse_model_name(MODEL_NAME)
SIM = hp_dict['SIM'] # TNG or Bolshoi
DEPTH = hp_dict['DEPTH']
FILTERS = hp_dict['FILTERS']
GRID = hp_dict['GRID']; SUBGRID = GRID//4; OFF = SUBGRID//2
LAMBDA_TH = hp_dict['LAMBDA_TH']
SIGMA = hp_dict['SIGMA']
BATCHNORM = hp_dict['BN']
DROP = hp_dict['DROP']
#UNIFORM_FLAG = hp_dict['UNIFORM_FLAG'] # i dont want to reset this
LOSS = hp_dict['LOSS'] # this messes w/ binary models, axe for now
if BINARY_MASK:
  LOSS = 'BCE'
model_TL_TYPE = hp_dict['TL_TYPE']
base_L = hp_dict['base_L']
DATE = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
if GRID == 640:
  SUBGRID = 128; OFF = 64
#================================================================
# set paths:
#================================================================
path_to_TNG = ROOT_DIR + 'data/TNG/'
path_to_BOL = ROOT_DIR + 'data/Bolshoi/'
if SIM == 'TNG':
  DATA_PATH = path_to_TNG
elif SIM == 'Bolshoi' or SIM == 'BOL':
  DATA_PATH = path_to_BOL
FIG_DIR_PATH = ROOT_DIR + 'figs/'
MODEL_PATH = ROOT_DIR + 'models/'
PRED_PATH = ROOT_DIR + 'preds/'
FILE_DEN = DATA_PATH + FN_DEN
FILE_MODEL = MODEL_PATH + MODEL_NAME
#================================================================
# Parse hp_dict file for attributes, set metrics
#================================================================
hp_dict_model = {}
hp_dict_model['MODEL_NAME_ATTRIBUTES'] = hp_dict
hp_dict_path = MODEL_PATH + MODEL_NAME + '_hps.txt'
try:
  hp_dict = nets.load_dict_from_text(hp_dict_path)
  REGULARIZE_FLAG = hp_dict['REGULARIZE_FLAG']
  hp_dict_model['BASE_MODEL_ATTRIBUTES'] = hp_dict
  if LOSS == 'FOCAL_CCE':
    alpha = hp_dict['focal_alpha']
    gamma = hp_dict['focal_gamma']
    alpha_list_float = ast.literal_eval(alpha)
    gamma = float(gamma)
  print('>>> Model hps found:',hp_dict_path)
except:
  print('>>> Model hps not found:',hp_dict_path)
  print('>>> But whatever!!!! IDC :)')
  REGULARIZE_FLAG = False
  # parse model name for alpha, gamma if loss is focal
  # this is janky but whatever saving the hps is messed up and i dont wanna fix it
  if LOSS == 'FOCAL_CCE':
    if 'BAL' in MODEL_NAME:
      alpha = [0.25,0.25,0.25,0.25]
      gamma = 2.0
    if 'BAL_HG' in MODEL_NAME:
      alpha = [0.25,0.25,0.25,0.25]
      gamma = 3.0
    if 'VOL' in MODEL_NAME:
      alpha = [0.65,0.25,0.15,0.1]
      gamma = 2.0
    if 'VOL_HG' in MODEL_NAME:
      alpha = [0.65,0.25,0.15,0.1]
      gamma = 3.0
    if 'VW' in MODEL_NAME:
      alpha = [0.6,0.5,0.25,0.25]
      gamma = 2.0
    if 'VW_HG' in MODEL_NAME:
      alpha = [0.6,0.5,0.25,0.25]
      gamma = 3.0
    alpha_list_float = alpha
ONE_HOT_FLAG = True # for compute metrics callback
metrics = ['accuracy']
if LOSS == 'CCE':
  loss = nets.CategoricalCrossentropy()
elif LOSS == 'SCCE':
  loss = nets.SparseCategoricalCrossentropy()
  ONE_HOT_FLAG = False
elif LOSS == 'DISCCE':
  loss = [nets.SCCE_Dice_loss]
  ONE_HOT_FLAG = False
elif LOSS == 'FOCAL_CCE':
  loss = [nets.categorical_focal_loss(alpha=alpha_list_float,gamma=gamma)] 
  #loss = nets.CategoricalFocalCrossentropy(alpha=alpha,gamma=gamma)
elif LOSS == 'BCE':
  loss = nets.BinaryCrossentropy()
  BINARY_MASK = True
more_metrics = [nets.MCC_keras(int_labels=~ONE_HOT_FLAG),nets.balanced_accuracy_keras(int_labels=~ONE_HOT_FLAG),
                nets.void_F1_keras(int_labels=~ONE_HOT_FLAG),nets.F1_micro_keras(int_labels=~ONE_HOT_FLAG)]
if not LOW_MEM_FLAG:
  more_metrics += [nets.recall_micro_keras(int_labels=~ONE_HOT_FLAG),
                   nets.precision_micro_keras(int_labels=~ONE_HOT_FLAG),
                   nets.true_wall_pred_as_void_keras(int_labels=~ONE_HOT_FLAG)]
metrics += more_metrics
# set up custom objects for loading model
custom_objects = {}
custom_objects['MCC'] = nets.MCC_keras(int_labels=~ONE_HOT_FLAG)
custom_objects['balanced_accuracy'] = nets.balanced_accuracy_keras(int_labels=~ONE_HOT_FLAG)
custom_objects['void_F1'] = nets.void_F1_keras(int_labels=~ONE_HOT_FLAG)
custom_objects['F1_micro'] = nets.F1_micro_keras(int_labels=~ONE_HOT_FLAG)
custom_objects['recall_micro'] = nets.recall_micro_keras(int_labels=~ONE_HOT_FLAG)
custom_objects['precision_micro'] = nets.precision_micro_keras(int_labels=~ONE_HOT_FLAG)
custom_objects['true_wall_pred_as_void'] = nets.true_wall_pred_as_void_keras(int_labels=~ONE_HOT_FLAG)
if LOSS == 'FOCAL_CCE':
  custom_objects['categorical_focal_loss_fixed'] = nets.categorical_focal_loss(alpha=alpha_list_float,gamma=gamma)
if LOSS == 'DISCCE':
  custom_objects['SCCE_Dice_loss'] = nets.SCCE_Dice_loss
# clear custom objects, metrics for binary models:
if BINARY_MASK:
  custom_objects = {}
  metrics = ['accuracy']
  print('>>> Binary mask model, clearing custom objects and metrics')
  N_CLASSES = 1
  class_labels = ['void','wall']
# print metrics:
print('>>> Metrics:')
for metric in metrics:
  print(str(metric))
print(LOSS)
# print custom objects:
print('>>> Custom Objects:')
for key, value in custom_objects.items():
  print(f'{key}: {value}')
#===============================================================
# Load data
#===============================================================
# parse transfer L from FN_DEN
if SIM == 'TNG':
  tran_L = int(FN_DEN.split('_L')[1].split('_')[0])
  X_PREFIX = f'{SIM}_L{tran_L}_Nm={GRID}'
  Y_PREFIX = f'{SIM}_Nm={GRID}'
  if LOSS == 'SCCE' or LOSS == 'DISCCE':
    Y_PREFIX += '_int'
  FILE_X_TRAIN = DATA_PATH + X_PREFIX + '_X_train.npy'
  FILE_Y_TRAIN = DATA_PATH + Y_PREFIX + '_Y_train.npy'
  FILE_X_TEST = DATA_PATH + X_PREFIX + '_X_test.npy'
  FILE_Y_TEST = DATA_PATH + Y_PREFIX + '_Y_test.npy'
  if BINARY_MASK:
    FILE_Y_TRAIN = path_to_TNG + Y_PREFIX + '_Y_train_binary.npy'
    FILE_Y_TEST  = path_to_TNG + Y_PREFIX + '_Y_test_binary.npy'
  FILE_MASK = DATA_PATH + f'TNG300-3-Dark-mask-Nm={GRID}-th={LAMBDA_TH}-sig={SIGMA}.fvol'
  FILE_FIG = FIG_DIR_PATH + 'TNG/'
elif SIM == 'Bolshoi':
  tran_L = int(FN_DEN.split('L=')[1].split('.0')[0])
  X_PREFIX = f'BOL_L{tran_L}_Nm={GRID}'
  Y_PREFIX = f'BOL_Nm={GRID}'
  if LOSS == 'SCCE' or LOSS == 'DISCCE':
    Y_PREFIX += '_int'
  FILE_X_TRAIN = DATA_PATH + X_PREFIX + '_X_train.npy'
  FILE_Y_TRAIN = DATA_PATH + Y_PREFIX + '_Y_train.npy'
  FILE_X_TEST = DATA_PATH + X_PREFIX + '_X_test.npy'
  FILE_Y_TEST = DATA_PATH + Y_PREFIX + '_Y_test.npy'
  FILE_MASK = DATA_PATH + f'Bolshoi_bolshoi.delta416_mask_Nm={GRID}_sig={SIGMA}_thresh={LAMBDA_TH}.fvol'
  FILE_FIG = FIG_DIR_PATH + 'Bolshoi/'
if not os.path.exists(FILE_FIG):
  os.makedirs(FILE_FIG)
# load data!!!
if LOAD_INTO_MEM:
  # load entire dataset into memory
  print('>>> Loading full train, val data into memory')
  if LOW_MEM_FLAG:
    features, labels = nets.load_dataset_all(FILE_DEN,FILE_MASK,SUBGRID)
  else:
    features, labels = nets.load_dataset_all_overlap(FILE_DEN,FILE_MASK,SUBGRID,OFF)
  if EXTRA_INPUTS is not None:
    print('>>> Loading extra inputs:',EXTRA_INPUTS)
    extra_input = nets.chunk_array(EXTRA_INPUTS, SUBGRID, scale=True)
    print('>>> Extra inputs loaded!'); print('Extra input shape:',extra_input.shape)
  print('>>> Data loaded!')
  print('Features shape:',features.shape)
  print('Labels shape:',labels.shape)
  # split into training and val sets:
  test_size = 0.2
  X_index = np.arange(0, features.shape[0])
  X_train, X_test, Y_train, Y_test = nets.train_test_split(X_index,labels,
                                                          test_size=test_size,
                                                          random_state=seed)
  if EXTRA_INPUTS is not None:
    extra_train = extra_input[X_train]; extra_test = extra_input[X_test]
    print('>>> Extra inputs split into training and validation sets')
    print('Extra train shape:',extra_train.shape)
    print('Extra test shape:',extra_test.shape)
  X_train = features[X_train]; X_test = features[X_test]
  if EXTRA_INPUTS is not None:
    print('>>> Concatenating extra inputs to training and validation sets')
    X_train = np.concatenate((X_train,extra_train),axis=-1)
    X_test = np.concatenate((X_test,extra_test),axis=-1)
  print('>>> Training and validation sets created')
  print('X_train shape: ',X_train.shape); print('Y_train shape: ',Y_train.shape)
  print('X_test shape: ',X_test.shape); print('Y_test shape: ',Y_test.shape)
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
  print(f'>>> Split into training ({(1-test_size)*100}%) and validation ({test_size*100}%) sets')
  print('X_train shape: ',X_train.shape); print('Y_train shape: ',Y_train.shape)
  print('X_test shape: ',X_test.shape); print('Y_test shape: ',Y_test.shape)
  if (LOSS != 'SCCE' and LOSS != 'DISCCE'):
    if not BINARY_MASK:
      print('>>> Converting to one-hot encoding')
      Y_train = nets.to_categorical(Y_train, num_classes=N_CLASSES,dtype='uint8')
      Y_test  = nets.to_categorical(Y_test, num_classes=N_CLASSES,dtype='uint8')
      print('>>> One-hot encoding complete')
      print('Y_train shape: ',Y_train.shape)
      print('Y_test shape: ',Y_test.shape)
  print('>>> Data loaded!')
  # Make tf.data.Dataset
  train_dataset = tf.data.Dataset.from_tensor_slices((X_train,Y_train))
  test_dataset = tf.data.Dataset.from_tensor_slices((X_test,Y_test))
else:
  # load data from saved files into tf.data.Dataset
  print('>>> Loading train, val data into tf.data.Dataset from memmapped .npy files')
  print('X_train:',FILE_X_TRAIN); print('Y_train:',FILE_Y_TRAIN)
  print('X_test:',FILE_X_TEST); print('Y_test:',FILE_Y_TEST)
  last_dim = 1 if LOSS == 'SCCE' or LOSS == 'DISCCE' else N_CLASSES
  if BINARY_MASK:
    last_dim = 1
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
# shuffle
print('>>> Shuffling and batching datasets')
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)
# batch and prefetch
print('>>> Prefetching datasets')
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
print('>>> Data loaded!')
print(f'Transfer learning on delta with L={tran_L}')
print('Density field:',FILE_DEN)
print('Mask field:',FILE_MASK)
# gonna skip saving val data bc I assume it is already...
#================================================================
# load and clone model
#================================================================
# check if FILE_MODEL exists, if not try with .keras extension
if not os.path.exists(FILE_MODEL):
  print('>>> Model not found:',FILE_MODEL)
  FILE_MODEL += '.keras'
  if not os.path.exists(FILE_MODEL):
    print('>>> Model not found:',FILE_MODEL)
    print('>>> Exiting...')
    sys.exit()
  else:
    print('>>> Model found:',FILE_MODEL)
# rename transfer learned model
CLONE_NAME = MODEL_NAME + '_TL_' + TL_TYPE + '_'
CLONE_NAME += 'tran_L'+str(tran_L)
if MULTI_FLAG:
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = nets.load_model(FILE_MODEL,custom_objects=custom_objects)
        clone = nets.clone_model(model)
        clone.set_weights(model.get_weights())
        clone._name = CLONE_NAME
        del model
        N_layers = len(clone.layers); print(f'Model has {N_layers} layers')
        if TL_TYPE == 'ENC':
          print('Freezing all layers up to bottleneck')
          first_up_name = f'decoder_block_D{DEPTH-1}_upsample'
          up_idx = nets.get_layer_index(clone,first_up_name) # up to 1st upsample layer
          print('Freezing all layers up to', clone.layers[up_idx].name)
          for layer in clone.layers[:up_idx]:
            layer.trainable = False
        elif TL_TYPE == 'LL':
          print('Freezing all layers up to last convolutional block')
          up_to_last_decode_idx = nets.get_layer_index(clone,'decoder_block_D0')
          up_to_last_decode_idx -= 2 # dont want to freeze that block, rather the one before!
          for layer in clone.layers[:up_to_last_decode_idx]:
            layer.trainable = False
        elif TL_TYPE == 'ENC_EO':
          # freeze every other block on the encoding side
          # blocks are named: enocder_block_D{i}
          # and encoder_block_D{i}_1, so we freeze the latter
          print('Freezing every other encoding block')
          for i in range(0,DEPTH):
            block_name = f'encoder_block_D{i}_1'
            block_idx = nets.get_layer_index(clone,block_name)
            freeze_blocks = []
            freeze_blocks.append(block_idx)
            print('Freezing',block_name)
          # get bottleneck index, freeze second conv
          bottle_idx = nets.get_layer_index(clone,'bottleneck_1')
          print('Freezing bottleneck_1')
          freeze_blocks.append(bottle_idx)
          for i in range(len(freeze_blocks)):
            layer = clone.layers[freeze_blocks[i]]
            layer.trainable = False
        elif TL_TYPE == 'ENC_D1':
          # freeze encoding side down to depth 2 (really 1 since it is zero indexed)
          # blocks are named: enocder_block_D{i}
          # freeze down to encoder_block_D1_maxpool
          print('Freezing encoding side down to depth 1 (really 2 since it is zero indexed)')
          block_name = 'encoder_block_D1_maxpool'
          block_idx = nets.get_layer_index(clone,block_name)
          print('Freezing all layers down to', clone.layers[block_idx].name)
          for layer in clone.layers[:block_idx]:
            layer.trainable = False
        elif TL_TYPE == 'ENC_D2':
          # freeze encoding side down to depth 3 (really 2 since it is zero indexed)
          # blocks are named: enocder_block_D{i}
          # freeze down to encoder_block_D2_maxpool
          print('Freezing encoding side down to depth 2 (really 3 since it is zero indexed)')
          block_name = 'encoder_block_D2_maxpool'
          block_idx = nets.get_layer_index(clone,block_name)
          print('Freezing all layers down to', clone.layers[block_idx].name)
          for layer in clone.layers[:block_idx]:
            layer.trainable = False
        # compile model:
        clone.compile(optimizer=nets.Adam(learning_rate=LR),loss=loss,
                      metrics=metrics)
else:
    model = nets.load_model(FILE_MODEL,custom_objects=custom_objects)
    clone = nets.clone_model(model)
    clone.set_weights(model.get_weights())
    clone._name = CLONE_NAME
    del model
    N_layers = len(clone.layers); print(f'Model has {N_layers} layers')
    if TL_TYPE == 'ENC':
      print('Freezing all layers up to bottleneck')
      first_up_name = f'decoder_block_D{DEPTH-1}_upsample'
      up_idx = nets.get_layer_index(clone,first_up_name) # up to 1st upsample layer
      print('Freezing all layers up to', clone.layers[up_idx].name)
      for layer in clone.layers[:up_idx]:
        layer.trainable = False
    elif TL_TYPE == 'LL':
      print('Freezing all layers up to last convolutional block')
      up_to_last_decode_idx = nets.get_layer_index(clone,'decoder_block_D0')
      up_to_last_decode_idx -= 2 # dont want to freeze that block, rather the one before!
      for layer in clone.layers[:up_to_last_decode_idx]:
        layer.trainable = False
    elif TL_TYPE == 'ENC_EO':
      # freeze every other block on the encoding side
      # blocks are named: enocder_block_D{i}
      # and encoder_block_D{i}_1, so we freeze the latter
      print('Freezing every other encoding block')
      for i in range(0,DEPTH):
        block_name = f'encoder_block_D{i}_1'
        block_idx = nets.get_layer_index(clone,block_name)
        freeze_blocks = []
        freeze_blocks.append(block_idx)
        print('Freezing',block_name)
      bottle_idx = nets.get_layer_index(clone,'bottleneck_1')
      print('Freezing bottleneck_1')
      freeze_blocks.append(bottle_idx)
      for i in range(len(freeze_blocks)):
        layer = clone.layers[freeze_blocks[i]]
        layer.trainable = False
    elif TL_TYPE == 'ENC_D1':
      # freeze encoding side down to depth 2 (really 1 since it is zero indexed)
      # blocks are named: enocder_block_D{i}
      # freeze down to encoder_block_D1_maxpool
      print('Freezing encoding side down to depth 1 (really 2 since it is zero indexed)')
      block_name = 'encoder_block_D1_maxpool'
      block_idx = nets.get_layer_index(clone,block_name)
      print('Freezing all layers down to', clone.layers[block_idx].name)
      for layer in clone.layers[:block_idx]:
        layer.trainable = False
    elif TL_TYPE == 'ENC_D2':
      # freeze encoding side down to depth 3 (really 2 since it is zero indexed)
      # blocks are named: enocder_block_D{i}
      # freeze down to encoder_block_D2_maxpool
      print('Freezing encoding side down to depth 2 (really 3 since it is zero indexed)')
      block_name = 'encoder_block_D2_maxpool'
      block_idx = nets.get_layer_index(clone,block_name)
      print('Freezing all layers down to', clone.layers[block_idx].name)
      for layer in clone.layers[:block_idx]:
        layer.trainable = False
    clone.compile(optimizer=nets.Adam(learning_rate=LR),loss=loss,
                  metrics=metrics)
clone.summary()
# save clone hps, WITH alpha and gamma if loss is focal:
hp_dict_model['FILE_MASK'] = FILE_MASK
hp_dict_model['BASE_MODEL'] = MODEL_NAME
hp_dict_model['MODEL_NAME'] = CLONE_NAME
hp_dict_model['TL_TYPE'] = TL_TYPE
if LOSS == 'FOCAL_CCE':
  hp_dict_model['focal_alpha'] = alpha
  hp_dict_model['focal_gamma'] = gamma
FILE_HPS_CLONE = FILE_MODEL+CLONE_NAME+'_hps.txt'
# get # of trainable params to ensure it's working:
trainable_ps = nets.layer_utils.count_params(clone.trainable_weights)
nontrainable_ps = nets.layer_utils.count_params(clone.non_trainable_weights)
hp_dict_model['trainable_params'] = trainable_ps
hp_dict_model['nontrainable_params'] = nontrainable_ps
hp_dict_model['total_params'] = trainable_ps + nontrainable_ps
nets.save_dict_to_text(hp_dict_model,FILE_HPS_CLONE)
#================================================================
# set up callbacks:
#================================================================
print('>>> Training')
# set up callbacks
model_chkpt = nets.ModelCheckpoint(MODEL_PATH+CLONE_NAME+'.keras',monitor='val_loss',
                                   save_best_only=True,verbose=2)
log_dir = ROOT_DIR + 'logs/fit/' + CLONE_NAME + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M") 
tb_call = nets.TensorBoard(log_dir=log_dir) # do we even need this if we CSV log?
csv_logger = nets.CSVLogger(MODEL_PATH+CLONE_NAME+'_' + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '_train_log.csv')
reduce_lr = nets.ReduceLROnPlateau(monitor='val_loss',factor=0.25,patience=lr_patience, 
                                   verbose=1,min_lr=1e-9)
early_stop = nets.EarlyStopping(monitor='val_loss',patience=patience,restore_best_weights=True)
callbacks = [model_chkpt,reduce_lr,early_stop,csv_logger,tb_call]
if TENSORBOARD_FLAG:
  callbacks.append(tb_call)
#===============================================================
# Train model
#===============================================================
history = clone.fit(train_dataset,epochs=epochs,validation_data=test_dataset,verbose=2,
                    callbacks=callbacks)
#================================================================
# plot performance metrics:
#================================================================
CLONE_FIG_DIR = FILE_FIG + CLONE_NAME + '/'
if not os.path.exists(CLONE_FIG_DIR):
  os.makedirs(CLONE_FIG_DIR)
  print('>>> Created directory for figures:',CLONE_FIG_DIR)
FILE_METRICS = CLONE_FIG_DIR + CLONE_NAME + '_metrics.png'
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
scores['L_TRAIN'] = base_L; scores['L_PRED'] = tran_L
scores['TL_TYPE'] = TL_TYPE
scores['UNIFORM_FLAG'] = UNIFORM_FLAG; scores['BATCHNORM'] = BATCHNORM
scores['DROPOUT'] = DROP; scores['LOSS'] = LOSS
scores['GRID'] = GRID; scores['DATE'] = DATE; scores['MODEL_NAME'] = CLONE_NAME
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
#===============================================================
# Predict, record metrics, and plot metrics on TEST DATA
#===============================================================
if LOAD_INTO_MEM:
  Y_pred = nets.run_predict_model(clone,X_test,batch_size,output_argmax=False)
else:
  Y_pred_list = []; Y_test_list = []
  for X_batch, Y_batch in test_dataset:
    Y_pred = clone.predict(X_batch,verbose=0)
    Y_pred_list.append(Y_pred)
    Y_test_list.append(Y_batch.numpy())
  Y_pred = np.concatenate(Y_pred_list,axis=0)
  Y_test = np.concatenate(Y_test_list,axis=0)
# since output argmax = False, Y_pred shape = [N_samples,SUBGRID,SUBGRID,SUBGRID,N_CLASSES]
# adjust Y_test shape to be [N_samples,SUBGRID,SUBGRID,SUBGRID,1]:
if (LOSS != 'SCCE' and LOSS != 'DISCCE'):
  if not BINARY_MASK:
    # undo one-hot encoding for input into save_scores_from_fvol
    Y_test = np.argmax(Y_test,axis=-1)
    Y_test = np.expand_dims(Y_test,axis=-1)
print('>>> Calculating scores on validation data')
if BINARY_MASK:
  N_CLASSES = 2 # janky fix for save_scores_from_fvol
nets.save_scores_from_fvol(Y_test,Y_pred,
                           MODEL_PATH+CLONE_NAME,CLONE_FIG_DIR,
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
if SIM == 'TNG':
  nets.save_scores_from_model(FILE_DEN, FILE_MASK, MODEL_PATH+CLONE_NAME, CLONE_FIG_DIR, PRED_PATH,
                              GRID=GRID,SUBGRID=SUBGRID,OFF=OFF,TRAIN_SCORE=False,
                              BINARY=BINARY_MASK)
elif SIM == 'BOL':
  nets.save_scores_from_model(FILE_DEN, FILE_MASK, MODEL_PATH+CLONE_NAME, CLONE_FIG_DIR, PRED_PATH,
                              GRID=GRID,SUBGRID=SUBGRID,OFF=OFF,BOXSIZE=256,BOLSHOI_FLAG=True,
                              TRAIN_SCORE=False,BINARY=BINARY_MASK)
print('>>> Finished predicting on training data')
#===============================================================
print('Finished training!')
print('Model name:',CLONE_NAME)
print('Interparticle spacing model trained on:',tran_L)
print(f'Model parameters: Depth={DEPTH}, Filters={FILTERS}, Uniform={UNIFORM_FLAG}, BatchNorm={BATCHNORM}, Dropout={DROP}')
print(f'Loss function: {LOSS}')
print('Date created:',DATE)
print('Total trainable parameters:',trainable_ps)
print('Total nontrainable parameters:',nontrainable_ps)
print('Total parameters:',trainable_ps+nontrainable_ps)
print('>>> Finished training!!!')
#===============================================================
# steps:
# score on test set
# save results to model_scores.csv 
# save slice plots of tran_L delta, pred, mask
# score on 45 deg rotated tran_L delta
# save scores with ortho_flag=False, val_flag=False
print('>>> Finished transfer learning!')