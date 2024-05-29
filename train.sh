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
  --FOCAL_ALPHA: Alpha value for focal loss. Default is 0.25. can be a sequence of 4 values.
  --FOCAL_GAMMA: Gamma value for focal loss. Default is 2.0.
  --LOAD_MODEL: If set to 1, load a previously trained model. Default is 0.
  --LOAD_INTO_MEM: If set to 1, load the entire dataset into memory. Default is 0.
  --BATCH_SIZE: Batch size for training. Default is 4.
  --EPOCHS: Number of epochs to train. Default is 500.
  --LEARNING_RATE: Learning rate for the optimizer. Default is 0.001.
  --REG_FLAG: If set to 1, use L2 regularization. Default is 0.
  --TENSORBOARD: If set, use TensorBoard. Default is to not.
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
F=32;  echo "Filters: $F";
LOSS="SCCE"; echo "Loss: $LOSS";
if [ "$SIM" = "TNG" ]; then
  GRID=512
elif [ "$SIM" = "BOL" ]; then
  GRID=640
fi
#GRID=256; echo "GRID: $GRID";

# optional flags initialization
BATCHNORM_ENABLED=0; echo "Batch Norm: $BATCHNORM_ENABLED";
DROPOUT_RATE=0.0; echo "Dropout: $DROPOUT_RATE";
MULTIPROCESSING_ENABLED=0; echo "Multiprocessing: $MULTIPROCESSING_ENABLED";
HIGH_MEM_ENABLED=0; echo "High memory usage: $HIGH_MEM_ENABLED";
FOCAL_ALPHA=(0.5 0.5 0.2 0.2); echo "Focal Alpha: ${FOCAL_ALPHA[@]}";
FOCAL_GAMMA=2.0; echo "Focal Gamma: $FOCAL_GAMMA";
UNIFORM_FLAG=0; echo "Uniform Flag: $UNIFORM_FLAG";
LOAD_MODEL=0; echo "Load Model: $LOAD_MODEL";
LOAD_INTO_MEM=0; echo "Load into memory: $LOAD_INTO_MEM";
BATCH_SIZE=4; echo "Batch Size: $BATCH_SIZE";
EPOCHS=500; echo "Epochs: $EPOCHS";
LEARNING_RATE=0.001; echo "Learning Rate: $LEARNING_RATE";
REG_FLAG=1; echo "Regularization: $REG_FLAG";
TENSORBOARD=0; echo "TensorBoard: $TENSORBOARD";

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
CMD_ARGS+=" --BATCH_SIZE $BATCH_SIZE"
CMD_ARGS+=" --EPOCHS $EPOCHS"
CMD_ARGS+=" --LEARNING_RATE $LEARNING_RATE"
[ "$REG_FLAG" -eq 1 ] && CMD_ARGS+=" --REGULARIZE_FLAG"
[ "$TENSORBOARD" -eq 1 ] && CMD_ARGS+=" --TENSORBOARD"
echo "Command line arguments: $CMD_ARGS";

# Running the Python script with dynamically constructed arguments
python3 ./deepvoid_misc/DV_MULTI_TRAIN.py $CMD_ARGS > ${output_fn} 2> ${error_fn};
kill $NVIDIA_SMI_PID