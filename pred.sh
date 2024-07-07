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
SIM="BOL"; echo "Simulation: $SIM";
BASE_L=0.122; echo "Base Lambda: $BASE_L";
D=4; echo "Depth: $D";
F=16; echo "Filters: $F";
LOSS="SCCE"; echo "Loss: $LOSS";
GRID=640; echo "Grid: $GRID";
TH=0.65; echo "Threshold: $TH";
TL_FLAG=1; echo "Transfer Flag: $TL_FLAG";
### CHOOSE LAMBDA TO PREDICT ON ###
TRAN_L=7; echo "Transfer/pred Lambda: $TRAN_L";
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
echo "Model Name: $MODEL_NAME";
echo "Mask Field: $FN_MSK";
echo "Density Field: $FN_DEN";

python3 ./deepvoid_misc/DV_MULTI_PRED.py $ROOT_DIR $SIM $MODEL_NAME $FN_DEN $FN_MSK $GRID > ${output_fn} 2> ${error_fn};
kill $NVIDIA_SMI_PID