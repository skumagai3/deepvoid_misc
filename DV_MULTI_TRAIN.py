#!/usr/bin/env python3
'''
3/17/24: Making an updated version of dv-train-nonbinary.py in nets.
I want to also make a lighter-weight version of the nets.py file,
which currently imports all kinds of stuff that I don't need.
'''
print('Running DV_MULTI_TRAIN.py')
import os
import sys
import datetime
import numpy as np
import tensorflow as tf
import NETS_LITE as nets
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
print('TensorFlow version: ', tf.__version__)
nets.K.set_image_data_format('channels_last')
#===============================================================
# Set random seed
#===============================================================
np.random.seed(12)
tf.random.set_seed(12)
#===============================================================
# Set parameters and paths
#===============================================================
class_labels = ['void','wall','fila','halo']
N_CLASSES = 4
### NOTE CHANGE THIS TO WHEREVER THE DATA IS STORED NOTE ###
# Picotte paths:
#path_to_data = '/ifs/groups/vogeleyGrp/data/TNG/' # path to TNG
#path_to_BOL = '/ifs/groups/vogeleyGrp/data/Bolshoi/' # path to Bolshoi
#FIG_DIR_PATH = '/ifs/groups/vogeleyGrp/nets/figs/' # path to figs save dir
#FILE_OUT = '/ifs/groups/vogeleyGrp/nets/models/' # path to models save dir
# Saturn paths:
path_to_data = '/Users/samkumagai/Desktop/deepvoid/simulations/IllustrisTNG/' # path to TNG
path_to_BOL = '/Users/samkumagai/Desktop/deepvoid/simulations/Bolshoi/' # path to Bolshoi
FIG_DIR_PATH = '/Users/samkumagai/Desktop/deepvoid/figs/' # path to figs save dir
FILE_OUT = '/Users/samkumagai/Desktop/deepvoid/model_weights/' # path to models save dir
#===============================================================
# arg parsing:
#===============================================================
if len(sys.argv) != 9:
  print('''Usage: python3 dv-train-nonbinary.py <SIM> <L> <DEPTH> <FILTERS> <UNIFORM_FLAG> <BATCHNORM> <DROPOUT> <LOSS>, 
        where GRID is the number of grid cells in the box, L is the interparticle separation in Mpc/h,
        DEPTH is the depth of the U-Net, FILTERS is the number of filters in the first layer,
        and UNIFORM_FLAG is 1 if you want to use identical masses for all subhaloes, 0 if not.
        BATCHNORM is 1 if you want to use batch normalization, 0 if not.
        DROPOUT is the dropout rate, and LOSS is the loss function to use.
        LOSS is one of 'CCE', 'FOCAL_CCE', 'DICE_AVG', or 'DICE_VOID'.''')
  sys.exit()
SIM = sys.argv[1]
L = sys.argv[2]
if L == '0.33' or L == '0.122':
  L = float(L) # since it's a float in the FULL DENSITY filename
else:
  L = int(L) # since it's an int in the SUBHALOES filename
DEPTH = int(sys.argv[3])
FILTERS = int(sys.argv[4])
UNIFORM_FLAG = bool(int(sys.argv[5]))
BATCHNORM = bool(int(sys.argv[6]))
DROPOUT = float(sys.argv[7])
LOSS = str(sys.argv[8])
print('>>> Parameters:')
print('Simulation = ', SIM); 
print('L = ',L); 
print('DEPTH = ',DEPTH); print('FILTERS = ',FILTERS)
print('UNIFORM_FLAG = ',UNIFORM_FLAG)
print('BATCHNORM = ',BATCHNORM)
print('DROPOUT = ',DROPOUT)
print('LOSS = ',LOSS)
print('#############################################')
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
  GRID = 512 
  SUBGRID = 128
  OFF = 64
  ### TNG Density field:
  # FULL (L=0.33 Mpc/h)
  FILE_DEN_FULL = path_to_data + f'DM_DEN_snap99_Nm={GRID}.fvol'
  if UNIFORM_FLAG == True:
    FILE_DEN_SUBS = path_to_data + f'subs1_mass_Nm{GRID}_L{L}_d_None_smooth_uniform.fvol'
  if UNIFORM_FLAG == False:
    FILE_DEN_SUBS = path_to_data + f'subs1_mass_Nm{GRID}_L{L}_d_None_smooth.fvol'
  if L <= 0.33:
    FILE_DEN = FILE_DEN_FULL
    L = 0.33 # full DM interparticle separation for TNG300-3-Dark
    UNIFORM_FLAG = False # doesnt affect full DM, so set to False
  else:
    FILE_DEN = FILE_DEN_SUBS
  ### Mask field:
  sig = 2.4 # PHI smooothing scale in code units NOTE CHANGES WITH NM
  FILE_MASK = path_to_data + f'TNG300-3-Dark-mask-Nm={GRID}-th={th}-sig={sig}.fvol'
  FILE_FIG = FIG_DIR_PATH + 'TNG_multiclass/'
  if not os.path.exists(FILE_FIG):
    os.makedirs(FILE_FIG)
### Bolshoi ###
elif SIM == 'BOL':
  GRID = 640
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
  FILE_FIG = FIG_DIR_PATH + 'Bolshoi_multiclass/'
  if not os.path.exists(FILE_FIG):
    os.makedirs(FILE_FIG)
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
features, labels = nets.load_dataset_all(FILE_DEN,FILE_MASK,SUBGRID,
                                         preproc='mm',binary_mask=False)
print('>>> Data loaded!'); print('Features shape: ',features.shape)
print('Labels shape: ',labels.shape)
# split into training and validation sets:
# X_train is the density subcubes used to train the model
# Y_train is the corresponding mask subcubes
# X_test is the density subcubes used to validate the model
# Y_test is the corresponding mask subcubes
X_train, X_test, Y_train, Y_test = nets.train_test_split(features,labels,test_size=0.2)
print('>>> Split into training and validation sets')
print('X_train shape: ',X_train.shape); print('Y_train shape: ',Y_train.shape)
print('X_test shape: ',X_test.shape); print('Y_test shape: ',Y_test.shape)
print('>>> Converting to one-hot encoding')
Y_train = nets.to_categorical(Y_train, num_classes=N_CLASSES)#,dtype='int8')
Y_test  = nets.to_categorical(Y_test, num_classes=N_CLASSES)#,dtype='int8')
print('>>> One-hot encoding complete')
#===============================================================
# Set model hyperparameters
#===============================================================
KERNEL = (3,3,3)
LR = 3e-4
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
DATE = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#===============================================================
# Save training and test arrays for later preds:
#===============================================================
if SIM == 'TNG':
  FILE_X_TEST = path_to_data + MODEL_NAME + '_X_test.npy'
  FILE_Y_TEST = path_to_data + MODEL_NAME + '_Y_test.npy'
if SIM == 'BOL':
  FILE_X_TEST = path_to_BOL + MODEL_NAME + '_X_test.npy'
  FILE_Y_TEST = path_to_BOL + MODEL_NAME + '_Y_test.npy'
SAVE_VAL_DATA = True # should only need to do once. make sure seed is set!
if SAVE_VAL_DATA:
  np.save(FILE_X_TEST,X_test,allow_pickle=True)
  np.save(FILE_Y_TEST,Y_test,allow_pickle=True)
  print(f'>>> Saved X_test to {FILE_X_TEST}')
  print(f'>>> Saved Y_test to {FILE_Y_TEST}')
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
hp_dict['BATCHNORM'] = str(BATCHNORM)
hp_dict['DROPOUT'] = str(DROPOUT)
hp_dict['DATE_CREATED'] = DATE
hp_dict['FILE_DEN'] = FILE_DEN
hp_dict['FILE_MASK'] = FILE_MASK
for key in hp_dict.keys():
  print(key,hp_dict[str(key)])
#===============================================================
# Set loss function
#===============================================================
if LOSS == 'CCE':
  loss = nets.CategoricalCrossentropy()
elif LOSS == 'FOCAL_CCE':
  loss = [nets.categorical_focal_loss(alpha=0.25,gamma=2.0)]
elif LOSS == 'DICE_AVG':
  # implement dice loss averaged over all classes
  pass
elif LOSS == 'DICE_VOID':
  # implement dice loss with void class
  pass
#===============================================================
# Multiprocessing
#===============================================================
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    model = nets.unet_3d((None,None,None,1),N_CLASSES,FILTERS,DEPTH,
                         batch_normalization=BATCHNORM,
                         dropout_rate=DROPOUT,
                         model_name=MODEL_NAME)
    model.compile(optimizer=nets.Adam(learning_rate=LR),
                                      loss=loss,
                                      metrics=['accuracy'])
model.summary()
# get trainable parameters:
trainable_ps = nets.count_params(model.trainable_weights)
nontrainable_ps = nets.count_params(model.non_trainable_weights)
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
epochs = 300; print('epochs: ',epochs)
patience = 50; print('patience: ',patience)
lr_patience = 25; print('learning rate patience: ',lr_patience)
batch_size = 8; print('batch_size: ',batch_size)
#===============================================================
print('>>> Training')
# set up callbacks
metrics = nets.ComputeMetrics((X_test,Y_test), N_epochs = 10, avg='macro')
model_chkpt = nets.ModelCheckpoint(FILE_OUT + MODEL_NAME, monitor='val_loss',
                                   save_best_only=True,verbose=1)
log_dir = "logs/fit/" + MODEL_NAME + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M") 
tb_call = nets.TensorBoard(log_dir=log_dir)
reduce_lr = nets.ReduceLROnPlateau(monitor='val_loss',factor=0.25,patience=lr_patience, 
                                   verbose=1,min_lr=1e-6)
early_stop = nets.EarlyStopping(monitor='val_loss',patience=patience,restore_best_weights=True)
callbacks = [metrics,model_chkpt,reduce_lr,tb_call,early_stop]
history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
                    validation_data=(X_test,Y_test), verbose = 2, shuffle = True,
                    callbacks=callbacks)
#===============================================================
# Check if figs directory exists, if not, create it:
#===============================================================
FIG_DIR = FILE_FIG + MODEL_NAME + '/'
if not os.path.exists(FIG_DIR):
  os.makedirs(FIG_DIR)
  print('>>> Created directory for figures: ',FIG_DIR)
# plot metrics here NOTE need new function for this...
#===============================================================
# Predict, record metrics, and plot metrics on TEST DATA
#===============================================================
Y_pred, Y_test = nets.run_predict_model(model,X_test,batch_size)
# adjust Y_test shape:
Y_test = np.argmax(Y_test,axis=-1); Y_test = np.expand_dims(Y_test,axis=-1)
#========================================================================
# Predict and plot and record metrics on TRAINING DATA
# with TRAIN_SCORE = False, all this does is predict on the entire 
# data cube and save slices of the predicted mask 
#========================================================================
if SIM == 'TNG':
  nets.save_scores_from_model(FILE_DEN, FILE_MASK, FILE_OUT+MODEL_NAME, FIG_DIR,
                              TRAIN_SCORE=False)
elif SIM == 'BOL':
  nets.save_scores_from_model(FILE_DEN, FILE_MASK, FILE_OUT+MODEL_NAME, FIG_DIR,
                              GRID=640,BOXSIZE=256,BOLSHOI_FLAG=True,TRAIN_SCORE=False)
print('>>> Finished predicting on training data')
#===============================================================
print('Finished training!')
print('Model name: ',MODEL_NAME)
print('Interparticle spacing model trained on: ',L)
print(f'Model parameters: Depth={DEPTH}, Filters={FILTERS}, Uniform={UNIFORM_FLAG}, BatchNorm={BATCHNORM}, Dropout={DROPOUT}')
print(f'Loss function: {LOSS}')
print('Date created: ',DATE)
print('Total trainable parameters: ',trainable_ps)
print('Total nontrainable parameters: ',nontrainable_ps)
print('Total parameters: ',trainable_ps+nontrainable_ps)
#===============================================================
