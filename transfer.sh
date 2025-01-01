#!/bin/bash
: <<'END_COMMENT'
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
        TL_TYPE: Transfer learning type. Example: 'ENC', 'LL', 'ENC_EO', 'ENC_D1',
        'ENC_D2'.

Optional Flags:
  --MULTI_FLAG: If set to 1, use multiprocessing. Default is 0.
  --LOW_MEM_FLAG: If set to 1, will load less training data and report fewer metrics. Default is 1.
  --LOAD_INTO_MEM: If set to 1, load the entire dataset into memory. Default is 0.
  --TENSORBOARD_FLAG: If set to 1, use TensorBoard. Default is 0.
  --EPOCHS: Number of epochs to train. Default is 500.
  --BATCH_SIZE: Batch size for training. Default is 8.
END_COMMENT
#######################################################################
ROOT_DIR="/content/drive/MyDrive/"; echo "Root directory: $ROOT_DIR";
current_time=$(date +"%Y%m%d-%H%M"); echo "Current time: $current_time";
mem_report="logs/GPU_usage/transfer_gpu_mem_usage_${current_time}.txt";
output="logs/stdout/transfer_output_${current_time}.txt";
error="logs/stderr/transfer_error_${current_time}.txt";
mem_report_fn="${ROOT_DIR}${mem_report}";
output_fn="${ROOT_DIR}${output}";
error_fn="${ROOT_DIR}${error}";
echo "Memory report file: $mem_report_fn";
echo "Output file: $output_fn";
echo "Error file: $error_fn";
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,memory.total,temperature.gpu,pstate --format=csv -l 30 > ${mem_report_fn} &
NVIDIA_SMI_PID=$!;
#######################################################################
### Select SIM: TNG/Bolshoi
SIM="TNG"; 
echo "Simulation: $SIM"; 
#######################################################################
# Choose model hyperparameters, choose base interparticle separation
# full dm density: TNG: 0.33, BOL: 0.122
# L = 3,5,7,10 for both
base_L=0.33; echo "Lambda: $base_L"; 
D=3; echo "Depth: $D";
F=32;  echo "Filters: $F";
LOSS="BCE"; echo "Loss: $LOSS"; # make blank if CCE
#######################################################################
### Interparticle separation for transfer learning
tran_L=10; echo "Transfer lambda: $tran_L";
TL_TYPE="ENC_EO"; echo "Transfer type: $TL_TYPE";
#######################################################################
if [ $SIM = "TNG" ]
then
    GRID=512; echo "GRID: $GRID"; ###### NOTE adjust grid here ########
    [ $GRID -eq 512 ] && SIGMA=2.4 || [ $GRID -eq 128 ] && SIGMA=0.6 || [ $GRID -eq 256 ] && SIGMA=1.2
    SIGMA=2.4; # 512 grid
    FN_DEN="subs1_mass_Nm${GRID}_L${tran_L}_d_None_smooth.fvol";
    SIM_PREFIX=$SIM;
elif [ $SIM = "Bolshoi" ] || [ $SIM = "BOL" ]
then
    SIGMA=0.916; # 640 grid
    GRID=640; echo "GRID: $GRID";
    FN_DEN="Bolshoi_halo_CIC_${GRID}_L=${tran_L}.0.fvol";
    SIM_PREFIX="Bolshoi";
fi
# other options (just type out model name manually):
BN=0; echo "Batch Norm: $BN";
MN_SUFFIX="BIN"; echo "Model Name Suffix: $MN_SUFFIX";
#UNIFORM_FLAG=0; echo "Uniform Flag: $UNIFORM_FLAG";
#DROP=0.0; echo "Dropout: $DROP";

# assuming L_th for every model is 0.65 (so far it is):
if [ $LOSS = "CCE" ] || [ $LOSS = "BCE"];
then
    if [ $BN -eq 1 ]
    then
        MODEL_NAME="${SIM_PREFIX}_D${D}-F${F}-Nm${GRID}-th0.65-sig${SIGMA}-base_L${base_L}_BN";
    else
        MODEL_NAME="${SIM_PREFIX}_D${D}-F${F}-Nm${GRID}-th0.65-sig${SIGMA}-base_L${base_L}";
    fi
else
    if [ $BN -eq 1 ]
    then
        MODEL_NAME="${SIM_PREFIX}_D${D}-F${F}-Nm${GRID}-th0.65-sig${SIGMA}-base_L${base_L}_BN_${LOSS}";
    else
        MODEL_NAME="${SIM_PREFIX}_D${D}-F${F}-Nm${GRID}-th0.65-sig${SIGMA}-base_L${base_L}_${LOSS}";
    fi
fi
# add model name suffix if not empty
if [ ! -z "$MN_SUFFIX" ];
then
    MODEL_NAME="${MODEL_NAME}_${MN_SUFFIX}";
fi
#######################################################################
# optional flags:
MULTI_FLAG=0; echo "Multiprocessing: $MULTI_FLAG"; # 0 for no, 1 for multiple GPUs
LOW_MEM_FLAG=0; echo "Low memory flag: $LOW_MEM_FLAG"; # 0 for no, 1 for yes
LOAD_INTO_MEM=0; echo "Load into memory: $LOAD_INTO_MEM"; # 0 for no, 1 for yes
TENSORBOARD_FLAG=0; echo "TensorBoard: $TENSORBOARD_FLAG"; # 0 for no, 1 for yes
EPOCHS=5; echo "Epochs: $EPOCHS";
BATCH_SIZE=8; echo "Batch Size: $BATCH_SIZE";
LEARNING_RATE=0.0003; echo "Learning Rate: $LEARNING_RATE";
LEARNING_RATE_PATIENCE=10; echo "Learning Rate Patience: $LEARNING_RATE_PATIENCE";
PATIENCE=25; echo "Patience: $PATIENCE";

# Constructing command line arguments dynamically
CMD_ARGS="$ROOT_DIR $MODEL_NAME $FN_DEN $TL_TYPE"
[ "$MULTI_FLAG" -eq 1 ] && CMD_ARGS+=" --MULTI_FLAG"
[ "$LOW_MEM_FLAG" -eq 1 ] && CMD_ARGS+=" --LOW_MEM_FLAG"
[ "$LOAD_INTO_MEM" -eq 1 ] && CMD_ARGS+=" --LOAD_INTO_MEM"
[ "$TENSORBOARD_FLAG" -eq 1 ] && CMD_ARGS+=" --TENSORBOARD_FLAG"
CMD_ARGS+=" --EPOCHS $EPOCHS"
CMD_ARGS+=" --BATCH_SIZE $BATCH_SIZE"
CMD_ARGS+=" --LEARNING_RATE $LEARNING_RATE"
CMD_ARGS+=" --LEARNING_RATE_PATIENCE $LEARNING_RATE_PATIENCE"
CMD_ARGS+=" --PATIENCE $PATIENCE"
echo "Command line arguments: $CMD_ARGS";

# Running the Python script with dynamically constructed arguments
python3 ./deepvoid_misc/DV_MULTI_TRANSFER.py $CMD_ARGS > ${output_fn} 2> ${error_fn};

kill $NVIDIA_SMI_PID