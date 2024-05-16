#!/usr/bin/env python3
'''
5/2/24: making a helper script to generate validation data.
otherwise, validation data is only generated during training.
This script is meant to be run on CPU, not GPU.

5/16/24: changing script to also save the training data.

We can save validation data ONLY because we set the random seed. 
As in every other script, we set seed = 12.
'''
import os
import sys
#import volumes
import numpy as np
from NETS_LITE import to_categorical, train_test_split, load_dataset_all_overlap, load_dataset_all
N_CLASSES = 4
#===============================================================
# Set random seed
#===============================================================
seed = 12
np.random.seed(seed)
#===============================================================
# arg parsing
# val data names are based on:
# ROOT_DIR
# SIM, L, GRID, and INT_FLAG.
# if INT_FLAG = True, then we are saving without one-hot encoding.
# if INT_FLAG = True, then the models will have SCCE loss
#===============================================================
if len(sys.argv) != 6:
    print('''Usage: python3 gen_val_data.py <ROOT_DIR> <SIM> <L>
    <GRID> <INT_FLAG>; where ROOT_DIR is the root dir where data,
    models, preds, figs, etc. are stored, SIM is either BOL or TNG,
    L is the interparticle separation, and GRID is the size of the cube
    on a side in voxels, and INT_FLAG indicates whether to save in 
    the one-hot encoded format or not (0 means one-hot, 1 means not). 
    ''')
    sys.exit()
ROOT_DIR = sys.argv[1]
SIM = sys.argv[2]
L = sys.argv[3]
if L == '0.33' or L == '0.122':
  L = float(L) # since it's a float in the FULL DENSITY filename
else:
  L = int(L) # since it's an int in the SUBHALOES filename
GRID = int(sys.argv[4])
INT_FLAG = bool(int(sys.argv[5]))
print('######################################')
print('>>> Running gen_val_data.py')
print('ROOT_DIR: ', ROOT_DIR)
print('SIM: ', SIM)
print('L: ', L)
print('GRID: ', GRID)
print('INT_FLAG: ', INT_FLAG)
print('######################################')
#===============================================================
# set paths
#===============================================================
path_to_TNG = ROOT_DIR + 'data/TNG/'
path_to_BOL = ROOT_DIR + 'data/Bolshoi/'
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
### TNG ### 
if SIM == 'TNG':
  #GRID = 512 
  SUBGRID = 128
  OFF = 64
  ### TNG Density field:
  # FULL (L=0.33 Mpc/h)
  FILE_DEN_FULL = path_to_TNG + f'DM_DEN_snap99_Nm={GRID}.fvol'
  FILE_DEN_SUBS = path_to_TNG + f'subs1_mass_Nm{GRID}_L{L}_d_None_smooth.fvol'
  if L == 0.33:
    FILE_DEN = FILE_DEN_FULL
  else:
    FILE_DEN = FILE_DEN_SUBS
  ### Mask field:
  sig = 2.4 # PHI smooothing scale in code units NOTE CHANGES WITH NM
  if GRID == 128:
    sig = 0.6
    SUBGRID = 32; OFF = 16
  FILE_MASK = path_to_TNG + f'TNG300-3-Dark-mask-Nm={GRID}-th={th}-sig={sig}.fvol'
### Bolshoi ###
elif SIM == 'BOL':
  #GRID = 640
  SUBGRID = 128
  OFF = 64
  if L == 0.122:
    FILE_DEN = path_to_BOL + f'Bolshoi_halo_CIC_{GRID}_L=0.122.fvol'
  else:
    FILE_DEN = path_to_BOL + f'Bolshoi_halo_CIC_{GRID}_L={L}.0.fvol'
  ### Mask field:
  sig = 0.916 # PHI smooothing scale in code units NOTE CHANGES WITH NM
  FILE_MASK = path_to_BOL + f'Bolshoi_bolshoi.delta416_mask_Nm={GRID}_sig={sig}_thresh={th}.fvol'
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
print('Density field: ',FILE_DEN)
print('Mask field: ',FILE_MASK)
features, labels = load_dataset_all_overlap(FILE_DEN,FILE_MASK,SUBGRID,OFF)
#features, labels = load_dataset_all(FILE_DEN,FILE_MASK,SUBGRID,OFF)
print('>>> Data loaded!'); print('Features shape: ',features.shape)
print('Labels shape: ',labels.shape)
# split into training and validation sets:
# X_train is the density subcubes used to train the model
# Y_train is the corresponding mask subcubes
# X_test is the density subcubes used to validate the model
# Y_test is the corresponding mask subcubes
test_size = 0.2
X_index = np.arange(0, features.shape[0])
X_train, X_test, Y_train, Y_test = train_test_split(X_index,labels,
                                                         test_size=test_size,
                                                         random_state=seed)
X_train = features[X_train]; X_test = features[X_test]
del features; del labels; del X_index # memory purposes
print(f'>>> Split into training ({(1-test_size)*100}%) and validation ({test_size*100}%) sets')
print('X_train shape: ',X_train.shape); print('Y_train shape: ',Y_train.shape)
print('X_test shape: ',X_test.shape); print('Y_test shape: ',Y_test.shape)
# print dtypes:
print(X_train.dtype); print(X_test.dtype)
print(Y_train.dtype); print(Y_test.dtype)
if INT_FLAG == False:
  print('>>> Converting to one-hot encoding')
  Y_train = to_categorical(Y_train, num_classes=N_CLASSES)#,dtype='int8')
  Y_test  = to_categorical(Y_test, num_classes=N_CLASSES)#,dtype='int8')
  print('>>> One-hot encoding complete')
  print('Y_train shape: ',Y_train.shape)
  print('Y_test shape: ',Y_test.shape)
# print shapes:
print(X_train.shape); print(X_test.shape)
print(Y_train.shape); print(Y_test.shape)
# print dtypes:
print(X_train.dtype); print(X_test.dtype)
print(Y_train.dtype); print(Y_test.dtype)
# print size in GB in memory:
print(X_train.nbytes/1e9); print(X_test.nbytes/1e9)
print(Y_train.nbytes/1e9); print(Y_test.nbytes/1e9)
#===============================================================
# Save training and test arrays for later preds:
# what parameters are important for the validation 
# data? SIM, L, GRID. 
# filenames: TNG: TNG_L{L}_GRID{GRID}_X_test.npy
# BOL: BOL_L{L}_GRID{GRID}_X_test.npy
# for Y_test though, the only things that matter is SIM, GRID,
# and if they're one-hot encoded or not. if LOSS = SCCE,
# add a flag that indicates the labels are integers.
#===============================================================
# save training set
# save X_train and Y_train to file:
X_TRAIN_DATA_NAME = f'{SIM}_L{L}_Nm={GRID}'
Y_TRAIN_DATA_NAME = f'{SIM}_Nm={GRID}'
if INT_FLAG == True:
    Y_TRAIN_DATA_NAME += '_int'
if SIM == 'TNG':
    FILE_X_TRAIN = path_to_TNG + X_TRAIN_DATA_NAME + '_X_train.npy'
    FILE_Y_TRAIN = path_to_TNG + Y_TRAIN_DATA_NAME + '_Y_train.npy'
if SIM == 'BOL':
    FILE_X_TRAIN = path_to_BOL + X_TRAIN_DATA_NAME + '_X_train.npy'
    FILE_Y_TRAIN = path_to_BOL + Y_TRAIN_DATA_NAME + '_Y_train.npy'
if os.path.exists(FILE_X_TRAIN) and os.path.exists(FILE_Y_TRAIN):
    print('>>> Training data already saved!')
elif os.path.exists(FILE_X_TRAIN) and not os.path.exists(FILE_Y_TRAIN):
    np.save(FILE_Y_TRAIN,Y_train,allow_pickle=True)
    print(f'File {FILE_X_TRAIN} already exists.')
    print(f'>>> Saved Y_train to {FILE_Y_TRAIN}')
elif not os.path.exists(FILE_X_TRAIN) and os.path.exists(FILE_Y_TRAIN):
    np.save(FILE_X_TRAIN,X_train,allow_pickle=True)
    print(f'File {FILE_Y_TRAIN} already exists.')
    print(f'>>> Saved X_train to {FILE_X_TRAIN}')
elif not os.path.exists(FILE_X_TRAIN) and not os.path.exists(FILE_Y_TRAIN):
    np.save(FILE_X_TRAIN,X_train,allow_pickle=True)
    np.save(FILE_Y_TRAIN,Y_train,allow_pickle=True)
    print(f'>>> Saved X_train to {FILE_X_TRAIN}')
    print(f'>>> Saved Y_train to {FILE_Y_TRAIN}')
# save test set
X_VAL_DATA_NAME = f'{SIM}_L{L}_Nm={GRID}'
Y_VAL_DATA_NAME = f'{SIM}_Nm={GRID}'
if INT_FLAG == True:
  Y_VAL_DATA_NAME += '_int'
if SIM == 'TNG':
  FILE_X_TEST = path_to_TNG + X_VAL_DATA_NAME + '_X_test.npy'
  FILE_Y_TEST = path_to_TNG + Y_VAL_DATA_NAME + '_Y_test.npy'
if SIM == 'BOL':
  FILE_X_TEST = path_to_BOL + X_VAL_DATA_NAME + '_X_test.npy'
  FILE_Y_TEST = path_to_BOL + Y_VAL_DATA_NAME + '_Y_test.npy'
if os.path.exists(FILE_X_TEST) and os.path.exists(FILE_Y_TEST):
  print('>>> Validation data already saved!')
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
print('>>> Validation data saved!')