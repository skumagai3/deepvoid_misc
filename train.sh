#!/bin/bash
ROOT_DIR="/Users/samkumagai/Desktop/Drexel/DeepVoid/"; echo "Root directory: $ROOT_DIR";
SIM="TNG"; echo "Simulation: $SIM";
L=0.33; echo "Lambda: $L";
D=2; echo "Depth: $D";
F=4;  echo "Filters: $F";
UNIFORM_FLAG=0; echo "Uniform Flag: $UNIFORM_FLAG";
BN=0; echo "Batch Norm: $BN";
DROP=0.0; echo "Dropout: $DROP";
LOSS="FOCAL_CCE"; echo "Loss: $LOSS";
MULTI_FLAG=0; echo "Multiprocessing: $MULTI_FLAG";
GRID=128; echo "GRID: $GRID";

python3 $ROOT_DIR/deepvoid_misc/DV_MULTI_TRAIN.py $ROOT_DIR $SIM $L $D $F $UNIFORM_FLAG $BN $DROP $LOSS $MULTI_FLAG $GRID;