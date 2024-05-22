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

SIM="TNG"; echo "Simulation: $SIM";
MODEL_NAME="TNG_D2-F4-Nm128-th0.65-sig0.6-base_L0.33_FOCAL"; echo "Model Name: $MODEL_NAME";
FN_DEN="DM_DEN_snap99_Nm=128.fvol"; echo "Density Field: $FN_DEN";
FN_MSK="TNG300-3-Dark-mask-Nm=128-th=0.65-sig=0.6.fvol"; echo "Mask Field: $FN_MSK";
GRID=128; echo "GRID: $GRID";

python3 ./deepvoid_misc/DV_MULTI_PRED.py $ROOT_DIR $SIM $MODEL_NAME $FN_DEN $FN_MSK $GRID > ${output_fn} 2> ${error_fn};
kill $NVIDIA_SMI_PID