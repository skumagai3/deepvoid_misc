#!/bin/bash
'''
Usage: ./transfer.sh

This script runs the DV_MULTI_TRANSFER.py script.

Parameters:
    Options that set the model hyperparameters and density file:
        base_L: Base interparticle separation.
        D: Depth of the model.
        F: Number of filters in the model.
        LOSS: Loss function to be used. Example: 'SCCE'.
        tran_L: Interparticle separation for transfer learning.
        SIM: Either 'TNG' or 'BOL'.
        GRID: Desired cube size on a side in voxels.
    Required arguments:
        ROOT_DIR: Root directory where data, models, figures, etc. are stored.
        MODEL_NAME: Name of the model.
        FN_DEN: Filename of the density field.
        TL_TYPE: Transfer learning type. Example: 'ENC'.

Optional Flags:
  --MULTI_FLAG: If set to 1, use multiprocessing. Default is 0.
  --LOW_MEM_FLAG: If set to 1, will load less training data and report fewer metrics. Default is 1.
'''
ROOT_DIR="/content/drive/MyDrive/"; echo "Root directory: $ROOT_DIR";
#######################################################################
# Choose model hyperparameters, choose base interparticle separation
# full dm density: TNG: 0.33, BOL: 0.122
# L = 3,5,7,10 for both
base_L=0.33; echo "Lambda: $base_L"; 
D=3; echo "Depth: $D";
F=16;  echo "Filters: $F";
LOSS="SCCE"; echo "Loss: $LOSS"; # make blank if CCE
#######################################################################
### Interparticle separation for transfer learning
tran_L=5; echo "Transfer lambda: $tran_L";
TL_TYPE="ENC"; echo "Transfer type: $TL_TYPE";
#######################################################################
### Select SIM: TNG/Bolshoi
SIM="TNG"; 
echo "Simulation: $SIM"; 
#######################################################################
if [ $SIM = "TNG" ]
then
    GRID=256; echo "GRID: $GRID"; ###### NOTE adjust grid here ########
    [ $GRID -eq 512 ] && SIGMA=2.4 || [ $GRID -eq 128 ] && SIGMA=0.6 || [ $GRID -eq 256 ] && SIGMA=1.2
    FN_DEN="subs1_mass_Nm${GRID}_L${tran_L}_d_None_smooth.fvol";
elif [ $SIM = "Bolshoi" ]
then
    SIGMA=0.916; # 640 grid
    GRID=640; echo "GRID: $GRID";
    FN_DEN="Bolshoi_halo_CIC_${GRID}_L=${tran_L}.0.fvol";
fi
# other options (just type out model name manually):
#UNIFORM_FLAG=0; echo "Uniform Flag: $UNIFORM_FLAG";
#BN=0; echo "Batch Norm: $BN";
#DROP=0.0; echo "Dropout: $DROP";

# assuming L_th for every model is 0.65 (so far it is):
if [ $LOSS = "CCE" ]
then
    MODEL_NAME="${SIM}_D${D}-F${F}-Nm${GRID}-th0.65-sig${SIGMA}-base_L${base_L}";
else
    MODEL_NAME="${SIM}_D${D}-F${F}-Nm${GRID}-th0.65-sig${SIGMA}-base_L${base_L}_${LOSS}";
fi
#######################################################################
# optional flags:
MULTI_FLAG=0; echo "Multiprocessing: $MULTI_FLAG"; # 0 for no, 1 for multiple GPUs
LOW_MEM_FLAG=1; echo "Low memory flag: $LOW_MEM_FLAG"; # 0 for no, 1 for yes

#python3 ./deepvoid_misc/DV_MULTI_TRANSFER.py $ROOT_DIR $MODEL_NAME $FN_DEN $TL_TYPE $MULTI_FLAG;
python3 $ROOT_DIR/deepvoid_misc/DV_MULTI_TRANSFER.py $ROOT_DIR $MODEL_NAME $FN_DEN $TL_TYPE --MULTI_FLAG $MULTI_FLAG --LOW_MEM_FLAG $LOW_MEM_FLAG;