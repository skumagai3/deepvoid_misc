#!/usr/bin/env python3
'''
4/24/24: SK
Script to evaluate models on validation data and plot slice plots from entire
cubes.
'''
print('>>> Running DV_MULTI_PRED.py')
import os
import sys
import numpy as np
import volumes
#import plotter
import absl.logging
import tensorflow as tf
import datetime
import NETS_LITE as nets
from scipy.ndimage import rotate
absl.logging.set_verbosity(absl.logging.ERROR)
print('TensorFlow version: ', tf.__version__)
nets.K.set_image_data_format('channels_last')
#===============================================================
# Set random seed
#===============================================================
seed = 12
np.random.seed(seed)
tf.random.set_seed(seed)
#===============================================================================
# arg parsing
#===============================================================================
if len(sys.argv) != 7:
    print('''Usage: python3 DV_MULTI_PRED.py <ROOT_DIR> <SIM> <MODEL_NAME>
          <FN_DEN> <FN_MSK> <GRID>; where ROOT_DIR is the root directory where data,
          models, preds, etc. are stored, SIM is either BOL or TNG, MODEL_NAME
          is the name of the network to load, and FN_DEN and FN_MSK are the 
          filepaths for the density and mask cubes respectively, GRID is the 
          desired cube size on a side in voxels.''')
    sys.exit()
ROOT_DIR = sys.argv[1]
SIM = sys.argv[2]
MODEL_NAME = sys.argv[3]
# NOTE that FN_DEN is not necessarily from the same sim as SIM
FN_DEN = sys.argv[4]
FN_MSK = sys.argv[5]
GRID = int(sys.argv[6])
DATE = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#===============================================================================
# parse MODEL_NAME for model attributes
#===============================================================================
DEPTH = int(MODEL_NAME.split('_D')[1][0]) 
FILTERS = int(MODEL_NAME.split('-F')[1].split('-')[0])
GRID_MODEL_NAME = int(MODEL_NAME.split('-Nm')[1].split('-')[0])
if GRID != GRID_MODEL_NAME:
    print(f'Model was trained on GRID size {GRID_MODEL_NAME} but you are using GRID size {GRID}.')
    sys.exit()
th = float(MODEL_NAME.split('-th')[1].split('-')[0])
if 'uniform' in MODEL_NAME:
    UNIFORM_FLAG = True
else:
    UNIFORM_FLAG = False
if 'BN' in MODEL_NAME:
    BATCHNORM = True
else:
    BATCHNORM = False
if 'DROP' in MODEL_NAME:
    DROPOUT = float(MODEL_NAME.split('DROP')[1].split('_')[0])
else:
    DROPOUT = 0.0 # AKA no dropout layers
LOSS = 'CCE' # default loss fxn
if 'FOCAL' in MODEL_NAME:
    LOSS = 'FOCAL_CCE'
if 'SCCE' in MODEL_NAME:
    LOSS = 'SCCE'
base_L = float(MODEL_NAME.split('base_L')[1].split('_')[0])
print('#############################################')
print('>>> Running DV_MULTI_PRED.py')
print('>>> Root directory: ',ROOT_DIR)
print('Simulation = ', SIM); 
print('Model trained on L = ',base_L); 
print('Model predicting on :',FN_DEN)
print('DEPTH = ',DEPTH); print('FILTERS = ',FILTERS)
print('Eigenvalue threshold used in mask = ',th)
print('UNIFORM_FLAG = ',UNIFORM_FLAG)
print('BATCHNORM = ',BATCHNORM)
print('DROPOUT = ',DROPOUT)
print('LOSS = ',LOSS)
print('GRID = ',GRID)
print('#############################################')
#===============================================================================
# set paths
#===============================================================================
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
#===============================================================================
# set parameters
#===============================================================================
SUBGRID = GRID//4; OFF = GRID//8 # for GRID=128 test case
if GRID == 512 or GRID == 640:
    SUBGRID = 128; OFF = 64
if SIM == 'TNG':
    BoxSize = 205
    BOLSHOI_FLAG = False
    DATA_PATH = path_to_TNG
    FILE_DEN = DATA_PATH + FN_DEN
    FILE_MSK = DATA_PATH + FN_MSK
    FIG_OUT = FIG_DIR_PATH + 'TNG/' + MODEL_NAME + '/'
if SIM == 'BOL':
    BoxSize = 256
    BOLSHOI_FLAG = True
    DATA_PATH = path_to_BOL
    FILE_DEN = DATA_PATH + FN_DEN
    FILE_MSK = DATA_PATH + FN_MSK
    FIG_OUT = FIG_DIR_PATH + 'Bolshoi/' + MODEL_NAME + '/'
# we want the figures to be saved in ROOT_DIR/figs/SIM/MODEL_NAME/:
if not os.path.exists(FIG_OUT):
    os.makedirs(FIG_OUT)
#===============================================================================
# load model (set compile=False if necessary?)
#===============================================================================
model = nets.load_model(FILE_OUT+MODEL_NAME)
model.summary()
#===============================================================================
# Check if validation set exists, if so, score on that
# if not, warn that these scores are based off data it was trained on
#===============================================================================
# we want to extract L from FILE_DEN...not necessarily base_L
if SIM == 'TNG':
    if FN_DEN == 'DM_DEN_snap99_Nm=512.fvol' or FN_DEN == 'DM_DEN_snap99_Nm=128.fvol':
        L = 0.33
    else:
        # recall TNG files have names like subs1_mass_Nm512_L3_d_None_smooth.fvol
        L = int(FN_DEN.split('L')[1].split('_')[0])
if SIM == 'BOL':
    if FN_DEN == 'Bolshoi_halo_CIC_640_L=0.122.fvol':
        L = 0.122
    else:
        # recall BOL files have names like Bolshoi_halo_CIC_640_L=5.0.fvol
        L = int(float(FN_DEN.split('L')[1].split('.fvol')[0]))
X_VAL_DATA_NAME = f'{SIM}_L{L}_Nm={GRID}'
Y_VAL_DATA_NAME = f'{SIM}_Nm={GRID}'
if LOSS == 'SCCE':
    Y_VAL_DATA_NAME += '_int'
X_TEST_PATH = DATA_PATH + X_VAL_DATA_NAME + '_X_test.npy'
Y_TEST_PATH = DATA_PATH + Y_VAL_DATA_NAME + '_Y_test.npy'
if os.path.exists(X_TEST_PATH) and os.path.exists(Y_TEST_PATH):
    VAL_FLAG = True
    X_test = np.load(X_TEST_PATH,allow_pickle=True)
    Y_test = np.load(Y_TEST_PATH,allow_pickle=True)
    print(f'Loaded validation features from {X_TEST_PATH}')
    print(f'Loaded validation labels from {Y_TEST_PATH}')
    # undo one-hot on Y_test if loss is not SCCE:
    if LOSS != 'SCCE':
        Y_test = np.argmax(Y_test,axis=-1)
        Y_test = np.expand_dims(Y_test,axis=-1)
else:
    VAL_FLAG = False
    print('Model is being scored on training data. Scores may be better than they actually should be.')
    X_test = nets.load_dataset(FILE_DEN, SUBGRID, OFF)
    Y_test = nets.load_dataset(FILE_MSK, SUBGRID, OFF, preproc=None, return_int=True)
#===============================================================================
# predict
# remember that Y_pred will have shape (N_samples, SUBGRID, SUBGRID, SUBGRID, 4)
#===============================================================================
batch_size = 8
Y_pred = nets.run_predict_model(model,X_test,batch_size,output_argmax=False)
#===============================================================================
# set up score_dict. 
# VAL_FLAG is True if scores are based on val set
# ORTHO_FLAG is True if scores are based on orthogonal rotated delta/mask
# for 45 deg rotated cubes, ORTHO_FLAG = False
#===============================================================================
scores = {}
ORTHO_FLAG = True
scores['SIM'] = SIM; scores['DEPTH'] = DEPTH; scores['FILTERS'] = FILTERS
scores['L_TRAIN'] = base_L; scores['L_PRED'] = L
scores['UNIFORM_FLAG'] = UNIFORM_FLAG; scores['BATCHNORM'] = BATCHNORM
scores['DROPOUT'] = DROPOUT; scores['LOSS'] = LOSS
scores['GRID'] = GRID; scores['DATE'] = DATE; scores['MODEL_NAME'] = MODEL_NAME
scores['VAL_FLAG'] = VAL_FLAG; scores['ORTHO_FLAG'] = ORTHO_FLAG
#===============================================================================
# score and save results to a row in model_scores.csv
#===============================================================================
nets.save_scores_from_fvol(Y_test,Y_pred,FILE_OUT+MODEL_NAME,
                           FIG_OUT,
                           scores,
                           VAL_FLAG)
nets.save_scores_to_csv(scores,ROOT_DIR+'model_scores.csv')
#===============================================================================
# plot slices from training data:
#===============================================================================
if SIM == 'TNG':
  nets.save_scores_from_model(FILE_DEN, FILE_MSK, FILE_OUT+MODEL_NAME, FIG_OUT, FILE_PRED,
                              GRID=GRID,SUBGRID=SUBGRID,OFF=OFF,TRAIN_SCORE=False)
elif SIM == 'BOL':
  nets.save_scores_from_model(FILE_DEN, FILE_MSK, FILE_OUT+MODEL_NAME, FIG_OUT, FILE_PRED,
                              GRID=GRID,SUBGRID=SUBGRID,OFF=OFF,BOXSIZE=256,BOLSHOI_FLAG=True,
                              TRAIN_SCORE=False)
#===============================================================================
# rotate training data (delta, mask) by 45 degrees and score again. 
# ORTHO_FLAG = False.... VAL_FLAG = False
# 45 rotated filepaths: path_to_TNG/ or path_to_BOL/ + '45deg/' + FN_DEN/FN_MSK
#===============================================================================
# check if rotated file exists
if os.path.exists(DATA_PATH+'45deg/'+FN_DEN) and os.path.exists(DATA_PATH+'45deg/'+FN_MSK):
    pass
else:
    # check if 45deg/ dir exists, if not create it.
    if not os.path.exists(DATA_PATH+'45deg/'):
        os.makedirs(DATA_PATH+'45deg/')
    print('45degree rotated files do not exist. Creating them...')
    d = volumes.read_fvolume(FILE_DEN); m = volumes.read_fvolume(FILE_MSK)
    d = rotate(d,45,reshape=False,mode='grid-wrap')
    m = rotate(m,45,reshape=False,mode='grid-wrap')
    print(f'Rotated density and mask for {FN_DEN} and {FN_MSK} by 45 degrees.')
    volumes.write_fvolume(d,DATA_PATH+'45deg/'+FN_DEN)
    volumes.write_fvolume(m,DATA_PATH+'45deg/'+FN_MSK)
# set up new score_dict with VAL_FLAG=False, ORTHO_FLAG=False
scores_45 = {}
ORTHO_FLAG = False; VAL_FLAG = False
scores['SIM'] = SIM; scores['DEPTH'] = DEPTH; scores['FILTERS'] = FILTERS
scores['L_TRAIN'] = base_L; scores['L_PRED'] = L
scores['UNIFORM_FLAG'] = UNIFORM_FLAG; scores['BATCHNORM'] = BATCHNORM
scores['DROPOUT'] = DROPOUT; scores['LOSS'] = LOSS
scores['GRID'] = GRID; scores['DATE'] = DATE; scores['MODEL_NAME'] = MODEL_NAME
scores['VAL_FLAG'] = VAL_FLAG; scores['ORTHO_FLAG'] = ORTHO_FLAG
# score
X_test = nets.load_dataset(DATA_PATH+'45deg/'+FN_DEN, SUBGRID, OFF)
Y_test = nets.load_dataset(DATA_PATH+'45deg/'+FN_MSK, SUBGRID, OFF, preproc=None, return_int=True)
Y_pred = nets.run_predict_model(model,X_test,batch_size,output_argmax=False)
# create FIG_OUT/45deg/ if it doesn't exist already:
if not os.path.exists(FIG_OUT+'45deg/'):
    os.makedirs(FIG_OUT+'45deg/')
nets.save_scores_from_fvol(Y_test,Y_pred,FILE_OUT+MODEL_NAME,
                           FIG_OUT+'45deg/',scores_45,VAL_FLAG)
# save to ROOT_DIR/model_scores.csv
nets.save_scores_to_csv(scores_45,ROOT_DIR+'model_scores.csv')
print('Saved 45 degree rotated scores!')
#===============================================================================
# plot slices from 45 degree rotated delta/mask.
#===============================================================================

print('>>> Finished predicting on training data')