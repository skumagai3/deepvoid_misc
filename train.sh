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
  LOSS: Loss function to be used. Default is CCE.
    Implemented losses:
      CCE: Categorical Crossentropy,
      FOCAL_CCE: Focal Categorical Crossentropy,
      SCCE: Sparse Categorical Crossentropy,
      DISCCE: Combo Dice and SCCE loss,
      SCCE_Void_Penalty: SCCE with a penalty for guessing the wrong proportion of voids,
      BCE: Binary Crossentropy.
  GRID: Desired cube size on a side in voxels. For TNG use 512, for BOL use 640.

Optional Flags:
  --UNIFORM_FLAG: If set to 1, use uniform mass subhaloes. Default is 0.
  --BATCHNORM: If set to 1, use batch normalization. Default is 0.
  --DROPOUT: Dropout rate. Default is 0.0, aka no dropout.
  --MULTI_FLAG: If set to 1, use multiprocessing. Default is 0.
  --LOW_MEM_FLAG: If set to 1, will load less training data and report fewer metrics. Default is 1.
  --FOCAL_ALPHA: Alpha value for focal loss. Default is 0.25. can be a sequence of 4 values.
  --FOCAL_GAMMA: Gamma value for focal loss. Default is 2.0.
  --MODEL_NAME_SUFFIX: Suffix to append to the model name. Default is empty.
  --LOAD_MODEL: If set to 1, load a previously trained model. Default is 0.
  --LOAD_INTO_MEM: If set to 1, load the entire dataset into memory. Default is 0.
  --BATCH_SIZE: Batch size for training. Default is 4.
  --EPOCHS: Number of epochs to train. Default is 500.
  --LEARNING_RATE: Learning rate for the optimizer. Default is 0.001.
  --LEARNING_RATE_PATIENCE: Patience for the learning rate scheduler. Default is 10.
  --PATIENCE: Patience for early stopping. Default is 25.
  --REG_FLAG: If set to 1, use L2 regularization. Default is 0.
  --PICOTTE_FLAG: If set to 1, use Picotte. Default is 0.
  --TENSORBOARD: If set, use TensorBoard. Default is to not.
  --BINARY_MASK: If set to 1, use binary mask. Default is 0. Requires BCE loss.
  --BOUNDARY_MASK: If set, uses a boundary mask. Default is to not.
  --EXTRA_INPUTS: If set, uses extra inputs. Default is to not.
  --ADD_RSD: If set to 1, adds RSD to the training data. Default is 0.
  --USE_PCONV: If set to 1, uses PConv. Default is 0.
  --ATTENTION_UNET: If set to 1, uses Attention UNet. Default is 0.
  --LAMBDA_CONDITIONING: If set to 1, uses lambda conditioning. Default is 0.
END_COMMENT
ROOT_DIR="/content/drive/MyDrive/"; echo "Root directory: $ROOT_DIR";
current_time=$(date +"%Y%m%d-%H%M"); echo "Current time: $current_time";
mem_report="logs/GPU_usage/train_gpu_mem_usage_${current_time}.txt";
output="logs/stdout/train_output_${current_time}.txt";
error="logs/stderr/train_error_${current_time}.txt";
mem_report_fn="${ROOT_DIR}${mem_report}";
output_fn="${ROOT_DIR}${output}";
error_fn="${ROOT_DIR}${error}";
echo "Memory report file: $mem_report_fn";
echo "Output file: $output_fn";
echo "Error file: $error_fn";
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,memory.total,temperature.gpu,pstate --format=csv -l 30 > ${mem_report_fn} &
NVIDIA_SMI_PID=$!;

SIM="TNG"; echo "Simulation: $SIM"; # TNG/BOL
L=0.33; echo "Lambda: $L";
D=3; echo "Depth: $D";
F=8;  echo "Filters: $F";
LOSS="SCCE"; echo "Loss: $LOSS";
if [ "$SIM" = "TNG" ]; then
  GRID=512
elif [ "$SIM" = "BOL" ] || [ "$SIM" = "Bolshoi"]; then
  GRID=640
fi
#GRID=256; echo "GRID: $GRID";

# optional flags initialization
BATCHNORM_ENABLED=1; echo "Batch Norm: $BATCHNORM_ENABLED";
DROPOUT_RATE=0.0; echo "Dropout: $DROPOUT_RATE";
MULTIPROCESSING_ENABLED=0; echo "Multiprocessing: $MULTIPROCESSING_ENABLED";
HIGH_MEM_ENABLED=0; echo "High memory usage: $HIGH_MEM_ENABLED";
FOCAL_ALPHA=(0.5 0.5 0.2 0.2); echo "Focal Alpha: ${FOCAL_ALPHA[@]}";
FOCAL_GAMMA=2.0; echo "Focal Gamma: $FOCAL_GAMMA";
MODEL_NAME_SUFFIX=""; echo "Model Name Suffix: $MODEL_NAME_SUFFIX";
UNIFORM_FLAG=0; echo "Uniform Flag: $UNIFORM_FLAG";
LOAD_MODEL=0; echo "Load Model: $LOAD_MODEL";
LOAD_INTO_MEM=1; echo "Load into memory: $LOAD_INTO_MEM";
BATCH_SIZE=4; echo "Batch Size: $BATCH_SIZE";
EPOCHS=2; echo "Epochs: $EPOCHS";
LEARNING_RATE=0.0001; echo "Learning Rate: $LEARNING_RATE";
LEARNING_RATE_PATIENCE=10; echo "Learning Rate Patience: $LEARNING_RATE_PATIENCE";
PATIENCE=25; echo "Patience: $PATIENCE";
REG_FLAG=0; echo "Regularization: $REG_FLAG";
PICOTTE_FLAG=0; echo "Picotte: $PICOTTE_FLAG";
TENSORBOARD=1; echo "TensorBoard: $TENSORBOARD";
BINARY_MASK=0; echo "Binary Mask: $BINARY_MASK";
ADD_RSD=1; echo "Add RSD: $ADD_RSD";
USE_PCONV=0; echo "Use PConv: $USE_PCONV";
ATTENTION_UNET=1; echo "Attention UNet: $ATTENTION_UNET";
LAMBDA_CONDITIONING=1; echo "Lambda Conditioning: $LAMBDA_CONDITIONING";

# Constructing command line arguments dynamically
CMD_ARGS="$ROOT_DIR $SIM $L $D $F $LOSS $GRID"
[ "$UNIFORM_FLAG" -eq 1 ] && CMD_ARGS+=" --UNIFORM_FLAG"
[ "$BATCHNORM_ENABLED" -eq 1 ] && CMD_ARGS+=" --BATCHNORM"
[ "$DROPOUT_RATE" != "0.0" ] && CMD_ARGS+=" --DROPOUT $DROPOUT_RATE"
[ "$MULTIPROCESSING_ENABLED" -eq 1 ] && CMD_ARGS+=" --MULTI_FLAG"
[ "$HIGH_MEM_ENABLED" -eq 1 ] && CMD_ARGS+=" --LOW_MEM_FLAG"
[ "$LOSS" = "FOCAL_CCE" ] && CMD_ARGS+=" --FOCAL_ALPHA ${FOCAL_ALPHA[@]} --FOCAL_GAMMA $FOCAL_GAMMA"
[ "$LOAD_MODEL" -eq 1 ] && CMD_ARGS+=" --LOAD_MODEL"
[ "$LOAD_INTO_MEM" -eq 1 ] && CMD_ARGS+=" --LOAD_INTO_MEM"
[ ! -z "$MODEL_NAME_SUFFIX" ] && CMD_ARGS+=" --MODEL_NAME_SUFFIX $MODEL_NAME_SUFFIX"
CMD_ARGS+=" --BATCH_SIZE $BATCH_SIZE"
CMD_ARGS+=" --EPOCHS $EPOCHS"
CMD_ARGS+=" --LEARNING_RATE $LEARNING_RATE"
CMD_ARGS+=" --LEARNING_RATE_PATIENCE $LEARNING_RATE_PATIENCE"
CMD_ARGS+=" --PATIENCE $PATIENCE"
[ "$REG_FLAG" -eq 1 ] && CMD_ARGS+=" --REGULARIZE_FLAG"
[ "$PICOTTE_FLAG" -eq 1 ] && CMD_ARGS+=" --PICOTTE_FLAG"
[ "$TENSORBOARD" -eq 1 ] && CMD_ARGS+=" --TENSORBOARD"
[ "$BINARY_MASK" -eq 1 ] && CMD_ARGS+=" --BINARY_MASK"
[ "$ADD_RSD" -eq 1 ] && CMD_ARGS+=" --ADD_RSD"
[ "$USE_PCONV" -eq 1 ] && CMD_ARGS+=" --USE_PCONV"
[ "$ATTENTION_UNET" -eq 1 ] && CMD_ARGS+=" --ATTENTION_UNET"
[ "$LAMBDA_CONDITIONING" -eq 1 ] && CMD_ARGS+=" --LAMBDA_CONDITIONING"
echo "Command line arguments: $CMD_ARGS";

# Running the Python script with dynamically constructed arguments
python3 ./deepvoid_misc/DV_MULTI_TRAIN.py $CMD_ARGS > ${output_fn} 2> ${error_fn};
kill $NVIDIA_SMI_PID