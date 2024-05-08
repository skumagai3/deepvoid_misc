#!/bin/bash
ROOT_DIR="/content/drive/MyDrive/"; echo "Root directory: $ROOT_DIR";
# full dm density: TNG: 0.33, BOL: 0.122
# L = 3,5,7,10 for both
base_L=0.33; echo "Lambda: $base_L"; 
D=4; echo "Depth: $D";
F=16;  echo "Filters: $F";
LOSS="SCCE"; echo "Loss: $LOSS"; # make blank if CCE

### Interparticle separation for transfer learning
tran_L=5; echo "Transfer lambda: $tran_L";
TL_TYPE="ENC"; echo "Transfer type: $TL_TYPE";

### Select SIM: TNG/Bolshoi
SIM="TNG"; 
echo "Simulation: $SIM"; 
if [ $SIM = "TNG" ]
then
    SIGMA=2.4; # 512 grid
    GRID=512; echo "GRID: $GRID"; # NOTE adjust grid here
    FN_DEN="subs1_mass_Nm${GRID}_L${tran_L}_d_None_smooth.fvol";
elif [ $SIM = "Bolshoi" ]
then
    SIGMA=0.916; # 640 grid
    GRID=640; echo "GRID: $GRID";
    FN_DEN="Bolshoi_halo_CIC_${GRID}_L=${tran_L}.0.fvol";
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
MULTI_FLAG=0; echo "Multiprocessing: $MULTI_FLAG"; # 0 for no, 1 for multiple GPUs

python3 ./deepvoid_misc/DV_MULTI_TRANSFER.py $ROOT_DIR $MODEL_NAME $FN_DEN $TL_TYPE $MULTI_FLAG;