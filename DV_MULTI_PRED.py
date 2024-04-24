#!/usr/bin/env python3
'''
4/24/24: SK
Script to evaluate models on validation data and plot slice plots from entire
cubes.
'''
import os
import sys
import nets
import volumes
import plotter
import NETS_LITE as nets
#===============================================================================
# arg parsing
#===============================================================================
if len(sys.argv) != 7:
    print('''Usage: python3 DV_MULTI_PRED.py <ROOT_DIR> <SIM> <MODEL_NAME>
          <FN_DEN> <FN_MSK>; where ROOT_DIR is the root directory where data,
          models, preds, etc. are stored, SIM is either BOL or TNG, MODEL_NAME
          is the name of the network to load, and FN_DEN and FN_MSK are the 
          filepaths for the density and mask cubes respectively, GRID is the 
          desired cube size on a side in voxels.''')
    sys.exit()
ROOT_DIR = sys.argv[1]
SIM = sys.argv[2]
MODEL_NAME = sys.argv[3]
FN_DEN = sys.argv[4]
FN_MSK = sys.argv[5]
GRID = int(sys.argv[6])
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
L = float(MODEL_NAME.split('base_L')[1].split('_')[0])
print('#############################################')
print('>>> Running DV_MULTI_PRED.py')
print('>>> Root directory: ',ROOT_DIR)
print('Simulation = ', SIM); 
print('L = ',L); 
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
    FILE_DEN = path_to_TNG + FN_DEN
    FILE_MSK = path_to_TNG + FN_MSK
    FIG_OUT = FIG_DIR_PATH + 'TNG/' + MODEL_NAME + '/'
if SIM == 'BOL':
    BoxSize = 256
    BOLSHOI_FLAG = True
    FILE_DEN = path_to_BOL + FN_DEN
    FILE_MSK = path_to_BOL + FN_MSK
    FIG_OUT = FIG_DIR_PATH + 'Bolshoi/' + MODEL_NAME + '/'

