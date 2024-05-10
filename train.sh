#!/bin/bash
: <<'END_COMMENT'
Usage: ./train.sh

This script runs the DV_MULTI_TRAIN.py script.

Parameters:
  ROOT_DIR: Root directory where data, models, figures, etc. are stored.
  SIM: Either 'TNG' or 'BOL'.
  L: Interparticle separation in Mpc/h. For TNG full DM use '0.33', 
  for BOL full DM use '0.122'. Other valid values are '3', '5', '7', '10'. 
  D: Depth of the model. Default is 3.
  F: Number of filters in the model. Default is 32.
  LOSS: Loss function to be used. CCE, FOCAL_CCE, or SCCE. Default is CCE.
  GRID: Desired cube size on a side in voxels. For TNG use 512, for BOL use 640.

Optional Flags:
  --UNIFORM_FLAG: If set to 1, use uniform mass subhaloes. Default is 0.
  --BATCHNORM: If set to 1, use batch normalization. Default is 0.
  --DROPOUT: Dropout rate. Default is 0.0, aka no dropout.
  --MULTI_FLAG: If set to 1, use multiprocessing. Default is 0.
  --LOW_MEM_FLAG: If set to 1, will load less training data and report fewer metrics. Default is 1.
END_COMMENT
ROOT_DIR="/content/drive/MyDrive/"; echo "Root directory: $ROOT_DIR";
SIM="TNG"; echo "Simulation: $SIM"; # TNG/BOL
L=0.33; echo "Lambda: $L";
D=3; echo "Depth: $D";
F=32;  echo "Filters: $F";
LOSS="SCCE"; echo "Loss: $LOSS";
GRID=256; echo "GRID: $GRID";
# optional
UNIFORM_FLAG=0; echo "Uniform Flag: $UNIFORM_FLAG";
BN=0; echo "Batch Norm: $BN";
DROP=0.0; echo "Dropout: $DROP";
MULTI_FLAG=0; echo "Multiprocessing: $MULTI_FLAG";
LOW_MEM_FLAG=1; echo "Low memory: $LOW_MEM_FLAG";

#python3 ./deepvoid_misc/DV_MULTI_TRAIN.py $ROOT_DIR $SIM $L $D $F $UNIFORM_FLAG $BN $DROP $LOSS $MULTI_FLAG $GRID;
python3 $ROOT_DIR/deepvoid_misc/DV_MULTI_TRAIN.py $ROOT_DIR $SIM $L $D $F $LOSS $GRID --UNIFORM_FLAG $UNIFORM_FLAG --BATCHNORM $BN --DROPOUT $DROP --MULTI_FLAG $MULTI_FLAG --LOW_MEM_FLAG $LOW_MEM_FLAG;