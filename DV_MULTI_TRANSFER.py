#!/usr/bin/env python3
'''
5/1/24: Making an updated version of dv-transfer-nonbinary.
NOTE that this is with the updated layer names.
Models' layers are named as such:
e.g. for a model with depth = 4:
input: InputLayer
depth=0 encoding block: encoder_block_D0
depth=0 maxpool layer: encoder_block_D0_maxpool

depth=1 encoding block: encoder_block_D1
depth=1 maxpool layer: encoder_block_D1_maxpool

depth=2 encoding block: encoder_block_D2
depth=2 maxpool layer: encoder_block_D2_maxpool

depth=3 encoding block: encoder_block_D3
depth=3 maxpool layer: encoder_block_D3_maxpool

depth=4 bottleneck: bottleneck

depth=3 upsampling layer: decoder_block_D3_upsample
depth=3 concatenation layer: decoder_block_D3_concat
depth=3 decoding block: decoder_block_D3

depth=2 upsampling layer: decoder_block_D2_upsample
depth=2 concatenation layer: decoder_block_D2_concat
depth=2 decoding block: decoder_block_D2

depth=1 upsampling layer: decoder_block_D1_upsample
depth=1 concatenation layer: decoder_block_D1_concat
depth=1 decoding block: decoder_block_D1

depth=0 upsampling layer: decoder_block_D0_upsample
depth=0 concatenation layer: decoder_block_D0_concat
depth=0 decoding block: decoder_block_D0

output: output_conv
'''
print('>>> Running DV_MULTI_TRANSFER.py')
import os
import sys
import datetime
import numpy as np
import tensorflow as tf
import NETS_LITE as nets
import absl.logging
import plotter
absl.logging.set_verbosity(absl.logging.ERROR)
print('TensorFlow version: ', tf.__version__)
nets.K.set_image_data_format('channels_last')
#===============================================================
# Set training parameters:
#===============================================================
# set to True if you want to use less memory, but no metrics and 
# less subcubes loaded into memory at once.
LOW_MEM_FLAG = True 
epochs = 200; print('epochs: ',epochs)
patience = 50; print('patience: ',patience)
lr_patience = 25; print('learning rate patience: ',lr_patience)
batch_size = 8; print('batch_size: ',batch_size)
N_epochs_metric = 10
print(f'classification metrics calculated every {N_epochs_metric} epochs')
KERNEL = (3,3,3)
LR = 3e-3 # increased to 3e-3 since we have LRreduceonplateau anyway
#===============================================================
# Set random seed
#===============================================================
seed = 12
np.random.seed(seed)
tf.random.set_seed(seed)
#===============================================================
# Set parameters and paths
#===============================================================
class_labels = ['void','wall','fila','halo']
N_CLASSES = 4
#===============================================================
# arg parsing:
#===============================================================
'''
What arguments will we need?
ROOT_DIR
MODEL_NAME (SIM, base_L will be pulled from that)
FN_DEN
TL_TYPE: (need to add more than just enc and LL)
- ENC: freeze entire encoding side (and bottleneck)
- LL: freeze entire model except last conv block and output

- ENC_D{freeze_depth}: freeze encoding side down to some depth?

What about double transfer learning?
What do i name those models?
'''
if len(sys.argv) != 6:
    print('''Usage: python3 DV_MULTI_TRANSFER.py <ROOT_DIR> <MODEL_NAME> <FN_DEN> <TL_TYPE> <MULTI_FLAG>;
          where ROOT_DIR is the root dir where data, models, preds, figs, etc. are stored,
          MODEL_NAME is the name of the model to be loaded, FN_DEN is the filename of the
          density field to be loaded, and TL_TYPE is the type of transfer learning to be
          done (e.g. ENC, LL, ENC_D3) MULTI_FLAG is 0 for no multiprocessing, 1 for multi.
          ''')
ROOT_DIR = sys.argv[1]
MODEL_NAME = sys.argv[2]
FN_DEN = sys.argv[3]
TL_TYPE = sys.argv[4]
MULTI_FLAG = bool(sys.argv[5])
# parse MODEL_NAME to get SIM, base_L, depth, filters, uniform_flag, BN, and drop
# MODEL_NAME format: {SIM}_D{depth}-F{filters}-Nm{GRID}-th0.65-sig{sig}-base_L{base_L}_{LOSS}
# e.g. TNG_D4-F16-Nm256-th0.65-sig1.2-base_L3_SCCE
hp_dict = nets.parse_model_name(MODEL_NAME)
SIM = hp_dict['SIM'] # TNG or Bolshoi
DEPTH = hp_dict['DEPTH']
FILTERS = hp_dict['FILTERS']
GRID = hp_dict['GRID']; SUBGRID = GRID//4; OFF = SUBGRID//2
LAMBDA_TH = hp_dict['LAMBDA_TH']
SIGMA = hp_dict['SIGMA']
BATCHNORM = hp_dict['BN']
DROP = hp_dict['DROP']
UNIFORM_FLAG = hp_dict['UNIFORM_FLAG']
LOSS = hp_dict['LOSS']
model_TL_TYPE = hp_dict['TL_TYPE']
base_L = hp_dict['base_L']
model_tran_L = hp_dict['tran_L']
DATE = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#================================================================
# set paths:
#================================================================
path_to_TNG = ROOT_DIR + 'data/TNG/'
path_to_BOL = ROOT_DIR + 'data/Bolshoi/'
if SIM == 'TNG':
    DATA_PATH = path_to_TNG
elif SIM == 'Bolshoi':
    DATA_PATH = path_to_BOL
FIG_DIR_PATH = ROOT_DIR + 'figs/'
MODEL_PATH = ROOT_DIR + 'models/'
PRED_PATH = ROOT_DIR + 'preds/'
FILE_DEN = DATA_PATH + FN_DEN
FILE_MODEL = MODEL_PATH + MODEL_NAME
#================================================================
# Parse hp_dict file for attributes
#================================================================
hp_dict_model = {}
with open(MODEL_PATH+MODEL_NAME+'_hps.txt','r') as f:
    for line in f:
        print(line)
        if line.split(':')[0] == 'FILE_MASK':
            break
        else:
            hp_dict[line.split(':')[0]] = line.split(':')[1].strip()
if LOSS == 'CCE':
    loss = nets.CategoricalCrossentropy()
elif LOSS == 'SCCE':
    loss = nets.SparseCategoricalCrossentropy()
elif LOSS == 'FOCAL_CCE':
    alpha = hp_dict_model['focal_alpha']
    gamma = hp_dict_model['focal_gamma']
    #loss = [nets.categorical_focal_loss(alpha=0.25,gamma=2.0)] 
    loss = nets.CategoricalFocalCrossentropy(alpha=alpha,gamma=gamma)
#===============================================================
# Load data
#===============================================================
# parse transfer L from FN_DEN
if SIM == 'TNG':
    tran_L = int(FN_DEN.split('_L')[1].split('_')[0])
    FILE_MASK = DATA_PATH + f'TNG300-3-Dark-mask-Nm={GRID}-th{LAMBDA_TH}-sig={SIGMA}.fvol'
    FILE_FIG = FIG_DIR_PATH + 'TNG/'
elif SIM == 'Bolshoi':
    tran_L = int(FN_DEN.split('L=')[1].split('.0')[0])
    FILE_MASK = DATA_PATH + f'Bolshoi_bolshoi.delta416_mask_Nm={GRID}_sig={SIGMA}_thresh={LAMBDA_TH}.fvol'
    FILE_FIG = FIG_DIR_PATH + 'Bolshoi/'
if not os.path.exists(FILE_FIG):
    os.makedirs(FILE_FIG)
print(f'Transfer learning on delta with L={tran_L}')
print('>>> Loading data!')
print('Density field:',FILE_DEN)
print('Mask field:',FILE_MASK)
if LOW_MEM_FLAG:
  features, labels = nets.load_dataset_all(FILE_DEN,FILE_MASK,SUBGRID)
else:
  features, labels = nets.load_dataset_all_overlap(FILE_DEN,FILE_MASK,SUBGRID,OFF)
print('>>> Data loaded!')
print('Features shape:',features.shape)
print('Labels shape:',labels.shape)
# split into training and val sets:
test_size = 0.2
X_index = np.arange(0, features.shape[0])
X_train, X_test, Y_train, Y_test = nets.train_test_split(X_index,labels,
                                                         test_size=test_size,
                                                         random_state=seed)
X_train = features[X_train]; X_test = features[X_test]
del features; del labels # memory purposes
print(f'>>> Split into training ({(1-test_size)*100}%) and validation ({test_size*100}%) sets')
print('X_train shape: ',X_train.shape); print('Y_train shape: ',Y_train.shape)
print('X_test shape: ',X_test.shape); print('Y_test shape: ',Y_test.shape)
if LOSS != 'SCCE':
  print('>>> Converting to one-hot encoding')
  Y_train = nets.to_categorical(Y_train, num_classes=N_CLASSES,dtype='uint8')
  Y_test  = nets.to_categorical(Y_test, num_classes=N_CLASSES,dtype='uint8')
  print('>>> One-hot encoding complete')
  print('Y_train shape: ',Y_train.shape)
  print('Y_test shape: ',Y_test.shape)
# gonna skip saving val data bc I assume it is already...
#================================================================
# load and clone model
#================================================================
# rename transfer learned model
CLONE_NAME = MODEL_NAME + '_TL_' + TL_TYPE + '_'
CLONE_NAME += 'tran_L'+str(tran_L)
if MULTI_FLAG:
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = nets.load_model(FILE_MODEL)
        clone = nets.clone_model(model)
        clone.set_weights(model.get_weights())
        clone._name = CLONE_NAME
        del model
        N_layers = len(clone.layers); print(f'Model has {N_layers} layers')
        if TL_TYPE == 'ENC':
            up_idx = nets.get_layer_index(clone,'up') # up to 1st upsample layer
            print('Freezing all layers up to', clone.layers[up_idx].name)
            for layer in clone.layers[:up_idx]:
                layer.trainable = False
        elif TL_TYPE == 'LL':
            print('Freezing all layers up to last convolutional block')
            up_to_last_decode_idx = nets.get_layer_index(clone,'decoder_block_D0')
            up_to_last_decode_idx -= 1 # dont want to freeze that block, rather the one before!
            for layer in clone.layers[:up_to_last_decode_idx]:
                layer.trainable = False
        # compile model:
        clone.compile(optimizer=nets.Adam(learning_rate=LR),loss=loss,
                      metrics=['accuracy'])
else:
    model = nets.load_model(FILE_MODEL)
    clone = nets.clone_model(model)
    clone.set_weights(model.get_weights())
    clone._name = CLONE_NAME
    del model
    N_layers = len(clone.layers); print(f'Model has {N_layers} layers')
    if TL_TYPE == 'ENC':
        up_idx = nets.get_layer_index(clone,'up') # up to 1st upsample layer
        print('Freezing all layers up to', clone.layers[up_idx].name)
        for layer in clone.layers[:up_idx]:
            layer.trainable = False
    elif TL_TYPE == 'LL':
        print('Freezing all layers up to last convolutional block')
        up_to_last_decode_idx = nets.get_layer_index(clone,'decoder_block_D0')
        up_to_last_decode_idx -= 1 # dont want to freeze that block, rather the one before!
        for layer in clone.layers[:up_to_last_decode_idx]:
            layer.trainable = False
    clone.compile(optimizer=nets.Adam(learning_rate=LR),loss=loss,
                  metrics=['accuracy'])
clone.summary()
# save clone hps, WITH alpha and gamma if loss is focal:
hp_dict_model['FILE_MASK'] = FILE_MASK
hp_dict_model['BASE_MODEL'] = MODEL_NAME
hp_dict_model['MODEL_NAME'] = CLONE_NAME
hp_dict_model['TL_TYPE'] = TL_TYPE
if LOSS == 'FOCAL_CCE':
    hp_dict_model['focal_alpha'] = alpha
    hp_dict_model['focal_gamma'] = gamma
FILE_HPS_CLONE = FILE_MODEL+CLONE_NAME+'_hps.txt'
# get # of trainable params to ensure it's working:
trainable_ps = nets.count_params(clone.trainable_weights)
nontrainable_ps = nets.count_params(clone.non_trainable_weights)
hp_dict_model['trainable_params'] = trainable_ps
hp_dict_model['nontrainable_params'] = nontrainable_ps
hp_dict_model['total_params'] = trainable_ps + nontrainable_ps
nets.save_dict_to_text(hp_dict_model,FILE_HPS_CLONE)
#================================================================
# set up callbacks:
#================================================================
print('>>> Training')
# set up callbacks
ONE_HOT_FLAG = True # for compute metrics callback
if LOSS == 'SCCE':
  ONE_HOT_FLAG = False
metrics = nets.ComputeMetrics((X_test,Y_test), N_epochs = N_epochs_metric, avg='micro', one_hot=ONE_HOT_FLAG)
model_chkpt = nets.ModelCheckpoint(FILE_OUT + CLONE_NAME, monitor='val_loss',
                                   save_best_only=True,verbose=2)
#log_dir = "logs/fit/" + MODEL_NAME + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M") 
#tb_call = nets.TensorBoard(log_dir=log_dir) # do we even need this if we CSV log?
csv_logger = nets.CSVLogger(FILE_OUT+CLONE_NAME+'_' + datetime.datetime.now().strftime("%Y%m%d-%H%M") + '_train_log.csv')
reduce_lr = nets.ReduceLROnPlateau(monitor='val_loss',factor=0.25,patience=lr_patience, 
                                   verbose=1,min_lr=1e-6)
early_stop = nets.EarlyStopping(monitor='val_loss',patience=patience,restore_best_weights=True)
if LOW_MEM_FLAG:
  # dont calc metrics, too memory intensive
  callbacks = [model_chkpt,reduce_lr,early_stop,csv_logger]
else:
  callbacks = [metrics,model_chkpt,reduce_lr,early_stop,csv_logger]
history = clone.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,
                    validation_data=(X_test,Y_test), verbose = 2, shuffle = True,
                    callbacks = callbacks)
#================================================================
# plot performance metrics:
#================================================================
CLONE_FIG_DIR = FILE_FIG + CLONE_NAME + '/'
if not os.path.exists(CLONE_FIG_DIR):
    os.makedirs(CLONE_FIG_DIR)
    print('>>> Created directory for figures:',CLONE_FIG_DIR)
FILE_METRICS = CLONE_FIG_DIR + CLONE_NAME + '_metrics.png'
plotter.plot_training_metrics_all(history,FILE_METRICS,savefig=True)
#===============================================================
# set up score_dict. 
# VAL_FLAG is True if scores are based on val set
# ORTHO_FLAG is True if scores are based on orthogonal rotated delta/mask
# for 45 deg rotated cubes, ORTHO_FLAG = False
#===============================================================
VAL_FLAG = True 
ORTHO_FLAG = True 
scores = {}
scores['SIM'] = SIM; scores['DEPTH'] = DEPTH; scores['FILTERS'] = FILTERS
scores['L_TRAIN'] = L; scores['L_PRED'] = L
scores['UNIFORM_FLAG'] = UNIFORM_FLAG; scores['BATCHNORM'] = BATCHNORM
scores['DROPOUT'] = DROP; scores['LOSS'] = LOSS
scores['GRID'] = GRID; scores['DATE'] = DATE; scores['MODEL_NAME'] = MODEL_NAME
scores['VAL_FLAG'] = VAL_FLAG
scores['ORTHO_FLAG'] = ORTHO_FLAG
epochs = len(history.epoch)
scores['EPOCHS'] = epochs
#===============================================================
# Predict, record metrics, and plot metrics on TEST DATA
#===============================================================
Y_pred = nets.run_predict_model(clone,X_test,batch_size,output_argmax=False)
# since output argmax = False, Y_pred shape = [N_samples,SUBGRID,SUBGRID,SUBGRID,N_CLASSES]
# adjust Y_test shape to be [N_samples,SUBGRID,SUBGRID,SUBGRID,1]:
if LOSS != 'SCCE':
  # undo one-hot encoding for input into save_scores_from_fvol
  Y_test = np.argmax(Y_test,axis=-1)
  Y_test = np.expand_dims(Y_test,axis=-1)
nets.save_scores_from_fvol(Y_test,Y_pred,
                           FILE_MODEL+CLONE_NAME,CLONE_FIG_DIR,
                           scores,
                           VAL_FLAG=VAL_FLAG)
# save score_dict by appending to the end of the csv.
# csv will be at ROOT_DIR/model_scores.csv
nets.save_scores_to_csv(scores,ROOT_DIR+'model_scores.csv')
#========================================================================
# Predict and plot and record metrics on TRAINING DATA
# with TRAIN_SCORE = False, all this does is predict on the entire 
# data cube and save slices of the predicted mask 
# for slice plotting:
#========================================================================
if SIM == 'TNG':
  nets.save_scores_from_model(FILE_DEN, FILE_MASK, FILE_OUT+MODEL_NAME, FIG_DIR, FILE_PRED,
                              GRID=GRID,SUBGRID=SUBGRID,OFF=OFF,TRAIN_SCORE=False)
elif SIM == 'BOL':
  nets.save_scores_from_model(FILE_DEN, FILE_MASK, FILE_OUT+MODEL_NAME, FIG_DIR, FILE_PRED,
                              GRID=GRID,SUBGRID=SUBGRID,OFF=OFF,BOXSIZE=256,BOLSHOI_FLAG=True,
                              TRAIN_SCORE=False)
print('>>> Finished predicting on training data')
#===============================================================
print('Finished training!')
print('Model name:',CLONE_NAME)
print('Interparticle spacing model trained on:',L)
print(f'Model parameters: Depth={DEPTH}, Filters={FILTERS}, Uniform={UNIFORM_FLAG}, BatchNorm={BATCHNORM}, Dropout={DROPOUT}')
print(f'Loss function: {LOSS}')
print('Date created:',DATE)
print('Total trainable parameters:',trainable_ps)
print('Total nontrainable parameters:',nontrainable_ps)
print('Total parameters:',trainable_ps+nontrainable_ps)
print('>>> Finished training!!!')
#===============================================================

# steps:
# score on test set
# save results to model_scores.csv 
# save slice plots of tran_L delta, pred, mask
# score on 45 deg rotated tran_L delta
# save scores with ortho_flag=False, val_flag=False
print('>>> Finished transfer learning!')