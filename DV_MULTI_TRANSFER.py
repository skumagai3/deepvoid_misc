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

- ENC_depth: freeze encoding side down to some depth?

What about double transfer learning?
What do i name those models?

'''


# steps:
# load and clone model
# freeze layers according to TL_TYPE
# create hyperparams txt file with TL details
# print out trainable/nontrainable params to ensure it's working
# load data, split into training/test sets
# perform transfer learning training
# plot training metrics
# score on test set
# save results to model_scores.csv 
# save slice plots of tran_L delta, pred, mask
# score on 45 deg rotated tran_L delta
# save scores with ortho_flag=False, val_flag=False
print('>>> Finished transfer learning!')