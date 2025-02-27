#!/bin/bash
: <<'END_COMMENT'
Usage: python3 DV_MULTI_PRED.py <ROOT_DIR> <SIM> <MODEL_NAME> <FN_DEN> <FN_MSK> <GRID>

This script makes predictions with a trained model.

Parameters:
  ROOT_DIR: Root directory where data, models, predictions, etc. are stored.
  SIM: Either 'BOL' or 'TNG'.
  MODEL_NAME: Name of the network to load.
  FN_DEN: Filepath for the density cube.
  FN_MSK: Filepath for the mask cube.
  GRID: Desired cube size on a side in voxels.
  TH: Threshold for the mask. always 0.65
  TL_FLAG: Transfer flag. 0 for base models, 1 for transfer learning models.
  Optional:
  XOVER_FLAG: set when performing crossover predictions from a model trained on one simulation to another.
  ORTHO_FLAG: set when performing preds on non-orthogonally rotated density fields, e.g. 45 degrees.
  CH4_FLAG: set when you want to save the 4-channel predictions (before argmax) to disk.
  BINARY_FLAG: set when you want to use a binary model for prediction.
  VAL_FLAG: set when you want to predict on the entire (training + val) set.
END_COMMENT
ROOT_DIR="/content/drive/MyDrive/"; echo "Root directory: $ROOT_DIR";
current_time=$(date +"%Y%m%d-%H%M"); echo "Current time: $current_time";
mem_report="logs/GPU_usage/pred_gpu_mem_usage_${current_time}.txt";
output="logs/stdout/pred_output_${current_time}.txt";
error="logs/stderr/pred_error_${current_time}.txt";
mem_report_fn="${ROOT_DIR}${mem_report}";
output_fn="${ROOT_DIR}${output}";
error_fn="${ROOT_DIR}${error}";
echo "Memory report file: $mem_report_fn";
echo "Output file: $output_fn";
echo "Error file: $error_fn";
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,memory.total,temperature.gpu,pstate --format=csv -l 30 > ${mem_report_fn} &
NVIDIA_SMI_PID=$!;

# BASE MODELS:
echo ">>> MODEL PARAMETERS:";
SIM="TNG"; echo "Simulation: $SIM";
BASE_L=0.33; echo "Base Lambda: $BASE_L";
D=3; echo "Depth: $D";
F=32; echo "Filters: $F";
LOSS="BCE"; echo "Loss: $LOSS";
GRID=512; echo "Grid: $GRID";
TH=0.65; echo "Threshold: $TH";
TL_FLAG=0; echo "Transfer Flag: $TL_FLAG";
### CHOOSE LAMBDA TO PREDICT ON ###
TRAN_L=7; echo "Transfer/pred Lambda: $TRAN_L";
### XOVER FLAG ###
XOVER_FLAG=0; echo "Crossover flag: $XOVER_FLAG";
### ORTHO FLAG ###
ORTHO_FLAG=1; echo "Orthogonal flag: $ORTHO_FLAG";
### SAVE 4 CHANNELS BEFORE ARGMAX ###
CH4_FLAG=1; echo "Save 4 channel prediction flag: $CH4_FLAG";
### BINARY MODEL FLAG ###
BINARY_FLAG=0; echo "Binary model flag: $BINARY_FLAG";
### VALIDATION FLAG ###
VAL_FLAG=0; echo "Validation flag: $VAL_FLAG";
#######################################################################
if [ "$TL_FLAG" = 0 ]; then
  TRAN_L=$BASE_L;
fi
if [ "$GRID" = "128" ]; then
  SIG=0.6;
elif [ "$GRID" = "256" ]; then
  SIG=1.2;
elif [ "$GRID" = "512" ]; then
  SIG=2.4;
elif [ "$GRID" = "640" ]; then
  SIG=0.916;
fi
echo "Sigma: $SIG";
# if loss is CCE, add nothing. if loss is FOCAL_CCE, add FOCAL. if loss is SCCE, add SCCE.
if [ "$LOSS" = "CCE" ]; then
  LOSS_SUFFIX=""
elif [ "$LOSS" = "FOCAL_CCE" ]; then
  LOSS_SUFFIX="FOCAL"
elif [ "$LOSS" = "SCCE" ]; then
  LOSS_SUFFIX="SCCE"
elif [ "$LOSS" = "BCE" ]; then
  LOSS_SUFFIX=""
  BINARY_FLAG=1
fi

### BASE MODELS ###
MODEL_NAME="_D${D}-F${F}-Nm${GRID}-th${TH}-sig${SIG}-base_L${BASE_L}_${LOSS_SUFFIX}";
### TL MODELS ###
if [ "$TL_FLAG" = 1 ]; then
  TL_TYPE="ENC_EO"; echo "Transfer type: $TL_TYPE";
  MODEL_NAME="_D${D}-F${F}-Nm${GRID}-th${TH}-sig${SIG}-base_L${BASE_L}_${LOSS_SUFFIX}_TL_${TL_TYPE}_tran_L${TRAN_L}";
fi
# set model and mask filenames:
if [ "$SIM" = "TNG" ]; then
  MODEL_NAME="TNG${MODEL_NAME}";
  FN_MSK="TNG300-3-Dark-mask-Nm=${GRID}-th=${TH}-sig=${SIG}.fvol";
  if [ "$TRAN_L" = "0.33" ]; then
    FN_DEN="DM_DEN_snap99_Nm=${GRID}.fvol"; # full TNG density
  else
    FN_DEN="subs1_mass_Nm${GRID}_L${TRAN_L}_d_None_smooth.fvol"; # subhalo TNG density
  fi
elif [ "$SIM" = "BOL" ]; then
  MODEL_NAME="Bolshoi${MODEL_NAME}";
  FN_MSK="Bolshoi_bolshoi.delta416_mask_Nm=${GRID}_sig=${SIG}_thresh=${TH}.fvol";
  if [ "$TRAN_L" = "0.122" ]; then
    FN_DEN="Bolshoi_halo_CIC_${GRID}_L=${TRAN_L}.fvol"; # full BOL density
  else
    FN_DEN="Bolshoi_halo_CIC_${GRID}_L=${TRAN_L}.0.fvol"; # full/subhalo BOL density
  fi
fi
MODEL_NAME="TNG_D3-F32-Nm512-th0.65-sig2.4-base_L0.33_BN_BIN";
echo "Model Name: $MODEL_NAME";
echo "Mask Field: $FN_MSK";
echo "Density Field: $FN_DEN";
echo "4 channel flag: $CH4_FLAG";
echo "Binary model? $BINARY_FLAG";
echo "Validation flag: $VAL_FLAG";
echo "Orthogonal flag: $ORTHO_FLAG";
echo "Xover flag: $XOVER_FLAG";

CMD_ARGS="$ROOT_DIR $SIM $MODEL_NAME $FN_DEN $FN_MSK $GRID";
[ "$XOVER_FLAG" -eq 1 ] && CMD_ARGS+=" --XOVER_FLAG";
[ "$ORTHO_FLAG" -eq 0 ] && CMD_ARGS+=" --ORTHO_FLAG";
[ "$CH4_FLAG" -eq 1 ] && CMD_ARGS+=" --CH4_FLAG";
[ "$BINARY_FLAG" -eq 1 ] && CMD_ARGS+=" --BINARY_FLAG";
[ "$VAL_FLAG" -eq 1 ] && CMD_ARGS+=" --VAL_FLAG";
echo "CMD_ARGS: $CMD_ARGS";

echo ">>> PREDICTING <<<";
python3 ./deepvoid_misc/DV_MULTI_PRED.py $CMD_ARGS > ${output_fn} 2> ${error_fn};
kill $NVIDIA_SMI_PID