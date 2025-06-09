'''
5/7/25:
Making a barebones version of DV_MULTI_TRAIN.py to test the effect
of boundary conditions on the performance of the model.
'''
print('>>> Running DV_BC_TRAIN.py')
import os
import sys
import time
import numpy as np
import tensorflow as tf
import NETS_LITE as nets
import absl.logging
import plotter
absl.logging.set_verbosity(absl.logging.ERROR)
print('Tensorflow version:', tf.__version__)
print('CUDA?', tf.test.is_built_with_cuda())
print('GPU?', tf.test.is_gpu_available())
nets.K.set_image_data_format('channels_last')
### Set the random seed for reproducibility
seed = 12; print('Setting random seed to', seed)
np.random.seed(seed)
tf.random.set_seed(seed)
class_labels = ['void','wall','filament','halo']
N_CLASSES = len(class_labels)
### Set paths 
DATA_PATH = '/ifs/groups/vogeleyGrp/data/TNG/bounded/'
MODEL_PATH = '/ifs/groups/vogeleyGrp/nets/models/TNG_boundary_test/'
FIGS_PATH = '/ifs/groups/vogeleyGrp/nets/figs/boundary_test/'
PRED_PATH = '/ifs/groups/vogeleyGrp/nets/preds/TNG_boundary_test/'

### Boundary conditions:
slab_start_idx = 200
slab_thickness = 64
bounds = f'slab_{slab_start_idx}_thickness_{slab_thickness}'

### Set attributes:
L = 0.33 # base TNG density
#L = 10 # Subhalo density
batch_size = 16

if L == 0.33:
    print('Using TNG base DM density')
    FILE_DEN = DATA_PATH + f'masked_TNG_DEN_BA_{bounds}.fvol'
if L == 10:
    print('Using TNG subhalo DM density')
    FILE_DEN = DATA_PATH + f'masked_TNG_DEN_10_{bounds}.fvol'
FILE_TRUTH = DATA_PATH + f'masked_TNG_TRUTH_{bounds}.fvol'
FILE_BOUND_MASK = DATA_PATH + f'boundary_mask_{bounds}.fvol'

### Data Loading:
features, labels = nets.load_dataset_all(
    FILE_DEN, FILE_TRUTH, 128,
    preproc=None
)
print('Features shape:', features.shape)
print('Labels shape:', labels.shape)
print('Features dtype:', features.dtype)
print('Labels dtype:', labels.dtype)
# same thing for the boundary mask:
features_bound, labels_bound = nets.load_dataset_all(
    FILE_BOUND_MASK, FILE_TRUTH, 128,
    preproc=None
)
print('Features shape:', features_bound.shape)
print('Labels shape:', labels_bound.shape)
print('Features dtype:', features_bound.dtype)
print('Labels dtype:', labels_bound.dtype)

test_size = 0.2
X_index = np.arange(0, features.shape[0])
X_train, X_test, Y_train, Y_test = nets.train_test_split(
    X_index, labels, test_size=test_size,
    random_state=seed   
)
X_train_bound, X_test_bound, Y_train_bound, Y_test_bound = nets.train_test_split(
    X_index, features_bound, test_size=test_size,
    random_state=seed   
)
X_train = features[X_train]; X_test = features[X_test]
del features, labels
X_train_bound = features_bound[X_train_bound]; X_test_bound = features_bound[X_test_bound]
del features_bound, labels_bound
# concatenate the two datasets:
X_train = np.concatenate((X_train, X_train_bound), axis=-1)
X_test = np.concatenate((X_test, X_test_bound), axis=-1)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('X_train dtype:', X_train.dtype)
print('X_test dtype:', X_test.dtype)

### Set up datasets:
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
print('>>> Shuffling and batching datasets')
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)
test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
print('>>> Prefetching datasets')
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
print('>>> Datasets ready')
### Set the model parameters
SIM = 'TNG'; print('Simulation:', SIM)
DEPTH = 3; print('Depth:', DEPTH)
FILTERS = 32; print('Filters:', FILTERS)
GRID = 512; print('Grid size:', GRID)
th = 0.65
sig = 2.4
L = 0.33 # intertracer spacing!!!
UNIFORM_FLAG = False
BATCH_NORM_FLAG = True
DROPOUT_FLAG = False
LOSS_FUNC = 'masked_CCE'
SUFFIX = f'boundary_test_{bounds}'

MODEL_NAME = nets.create_model_name(
    SIM, DEPTH, FILTERS, GRID, th, sig, L,
    UNIFORM_FLAG, BATCH_NORM_FLAG, DROPOUT_FLAG,
    LOSS_FUNC, suffix=SUFFIX
)
print('Model name:', MODEL_NAME)
