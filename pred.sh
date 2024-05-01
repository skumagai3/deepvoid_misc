#!/bin/bash
ROOT_DIR="/Users/samkumagai/Desktop/Drexel/DeepVoid/"; echo "Root directory: $ROOT_DIR";
SIM="TNG"; echo "Simulation: $SIM";
MODEL_NAME="TNG_D2-F4-Nm128-th0.65-sig0.6-base_L0.33_FOCAL"; echo "Model Name: $MODEL_NAME";
FN_DEN="DM_DEN_snap99_Nm=128.fvol"; echo "Density Field: $FN_DEN";
FN_MSK="TNG300-3-Dark-mask-Nm=128-th=0.65-sig=0.6.fvol"; echo "Mask Field: $FN_MSK";
GRID=128; echo "GRID: $GRID";

python3 $ROOT_DIR/deepvoid_misc/DV_MULTI_PRED.py $ROOT_DIR $SIM $MODEL_NAME $FN_DEN $FN_MSK $GRID;