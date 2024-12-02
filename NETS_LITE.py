#!/usr/bin/env python3
'''
3/17/24: Making an updated version of the nets.py file.
Importing less random stuff and making it more lightweight and readable.

This is meant to be used for multi-class classification.
The old nets.py is fine for binary.
'''
import re
import gc
import os
import sys
import csv
import volumes
import plotter
import pandas as pd
import tensorflow as tf
from collections import Counter
import matplotlib.pyplot as plt
class_labels = ['Void','Wall','Filament','Halo']
import numpy as np
from scipy import ndimage as ndi
from keras.models import Model, load_model, clone_model
from tensorflow.keras.optimizers import Adam # type: ignore
#from tensorflow.keras.utils.layer_utils import count_params # doesnt work?
from tensorflow.python.keras.utils import layer_utils
from keras.utils import to_categorical
from keras.layers import Input, Conv3D, MaxPooling3D, Conv3DTranspose, UpSampling3D, Concatenate, BatchNormalization, Activation, Dropout
from keras import backend as K
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy, BinaryCrossentropy
#from keras.losses import CategoricalFocalCrossentropy # not available in tf 2.10.0!!!
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
#from keras.saving import register_keras_serializable
from keras import regularizers
from tensorflow.keras import metrics
from concurrent.futures import ThreadPoolExecutor

from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, average_precision_score, auc, precision_recall_curve
from sklearn.metrics import RocCurveDisplay, precision_recall_curve, classification_report, precision_recall_fscore_support
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay
#---------------------------------------------------------
# Summary statistics for a Numpy array
#---------------------------------------------------------
def summary(array):
  print('### Summary Statistics ###')
  print('Shape: ',str(array.shape))
  print('Mean: ',np.mean(array))
  print('Median: ',np.median(array))
  print('Maximum: ',np.max(array))
  print('Minimum: ',np.min(array))
  print('Std deviation: ',np.std(array))
  print('Variance: ',np.var(array))
#---------------------------------------------------------
# Regularization options: minmax and standardize
#---------------------------------------------------------
def minmax(a):
    return (a-np.min(a))/(np.max(a)-np.min(a))
def standardize(a):
  return (a-np.mean(a))/(np.std(a))
#---------------------------------------------------------
# Convert multiclass mask to binary void/not void mask
#---------------------------------------------------------
def convert_to_binary_mask(mask):
  # Create a binary mask where class 0 remains 0 and 
  # classes 1, 2, 3 become 1. assuming mask is not one-hot.
  binary_mask = (mask > 0).astype(int)
  return binary_mask
#---------------------------------------------------------
# Assemble cube from subcubes
#---------------------------------------------------------
def assemble_cube2(Y_pred,GRID,SUBGRID,OFF):
    cube  = np.zeros(shape=(GRID,GRID,GRID))
    #nbins = (GRID // SUBGRID) + 1 + 1 + 1
    nbins = (GRID // SUBGRID) + (GRID // SUBGRID - 1)
    #if GRID == 640:
    #  nbins += 1
    cont  = 0
    
    SUBGRID_4 = SUBGRID//4
    SUBGRID_2 = SUBGRID//2
    
    for i in range(nbins):
        if i==0:
            di_0 = SUBGRID*i - OFF*i
            di_1 = SUBGRID*i - OFF*i + SUBGRID_4+SUBGRID_2
            si_0 =  0
            si_1 = -SUBGRID_4
        else:
            di_0 = SUBGRID*i - OFF*i + SUBGRID_4
            di_1 = SUBGRID*i - OFF*i + SUBGRID_4+SUBGRID_2
            si_0 =  SUBGRID_4
            si_1 = -SUBGRID_4            
            if i==nbins-1:
                di_0 = SUBGRID*i - OFF*i + SUBGRID_4
                di_1 = SUBGRID*i - OFF*i + SUBGRID
                si_0 =  SUBGRID_4
                si_1 =  SUBGRID

        for j in range(nbins):
            if j==0:
                dj_0 = SUBGRID*j - OFF*j
                dj_1 = SUBGRID*j - OFF*j + SUBGRID_4+SUBGRID_2
                sj_0 =  0
                sj_1 = -SUBGRID_4
            else:
                dj_0 = SUBGRID*j - OFF*j + SUBGRID_4
                dj_1 = SUBGRID*j - OFF*j + SUBGRID_4+SUBGRID_2
                sj_0 = SUBGRID_4
                sj_1 = -SUBGRID_4
                if j==nbins-1:
                    dj_0 = SUBGRID*j - OFF*j + SUBGRID_4
                    dj_1 = SUBGRID*j - OFF*j + SUBGRID
                    sj_0 = SUBGRID_4
                    sj_1 = SUBGRID                     
            for k in range(nbins):
                if k==0:
                    dk_0 = SUBGRID*k - OFF*k
                    dk_1 = SUBGRID*k - OFF*k + SUBGRID_4+SUBGRID_2
                    sk_0 =  0
                    sk_1 = -SUBGRID_4
                else:
                    dk_0 = SUBGRID*k - OFF*k + SUBGRID_4
                    dk_1 = SUBGRID*k - OFF*k + SUBGRID_4+SUBGRID_2
                    sk_0 =  SUBGRID_4
                    sk_1 = -SUBGRID_4
                    if k==nbins-1:
                        dk_0 = SUBGRID*k - OFF*k + SUBGRID_4
                        dk_1 = SUBGRID*k - OFF*k + SUBGRID
                        sk_0 = SUBGRID_4
                        sk_1 = SUBGRID                                                                                                        
                    
                cube[di_0:di_1, dj_0:dj_1, dk_0:dk_1] = Y_pred[cont, si_0:si_1, sj_0:sj_1, sk_0:sk_1,0]
                cont = cont+1
    return cube
#---------------------------------------------------------
# For loading training and testing data for training
# if loading data for regression, ensure classification=False!!
#---------------------------------------------------------
def load_dataset_all(FILE_DEN, FILE_MASK, SUBGRID, preproc='mm', classification=True, sigma=None, binary_mask=False):
  '''
  Function that loads the density and mask files, splits into subcubes of size
  SUBGRID, rotates by 90 degrees three times, and returns the X and Y data.
  FILE_DEN: str filepath to density field.
  FILE_MASK: str filepath to mask field.
  SUBGRID: int size of subcubes.
  preproc: str preprocessing method. 'mm' for minmax, 'std' for standardize.
  classification: bool whether or not you're doing classification. def True.
  sigma: float sigma for Gaussian smoothing. def None.
  binary_mask: bool whether or not to convert mask to binary. def False. 
  '''
  print(f'Reading volume: {FILE_DEN}... ')
  den = volumes.read_fvolume(FILE_DEN)
  if sigma is not None:
    den = ndi.gaussian_filter(den,sigma,mode='wrap')
    print(f'Smoothed density with a Gaussian kernel of size {sigma}')
  print(f'Reading mask: {FILE_MASK}...')
  msk = volumes.read_fvolume(FILE_MASK)
  # print mask populations:
  _, cnts = np.unique(msk,return_counts=True)
  for val in cnts:
    print(f'% of population: {val/den.shape[0]**3 * 100}')
  den_shp = den.shape
  #msk_shp = msk.shape
  summary(den); summary(msk)
  # binary mask oneliner
  if binary_mask == True:
    msk = (msk < 1.).astype(int)
    print('Converted mask to binary mask. Void = 1, not void = 0.')
    summary(den); summary(msk)
  if preproc == 'mm':
    #den = minmax(np.log10(den)) # this can create NaNs be careful
    den = minmax(den)
    #msk = minmax(msk) # 12/5 needed to disable this for sparse CCE losses
    print('Ran preprocessing to scale density to [0,1]!')
    print('\nNew summary statistics: ')
    summary(den)
  if preproc == 'std':
    den = standardize(den)
    #msk = standardize(msk)
    print('Ran preprocessing by dividing density/mask by std dev and subtracting by the mean! ')
    print('\nNew summary statistics: ')
    summary(den)
  # Make wall mask
  #msk = np.zeros(den_shp,dtype=np.uint8)
  n_bins = den_shp[0] // SUBGRID

  cont = 0 
  X_all = np.zeros(shape=((n_bins**3)*4, SUBGRID,SUBGRID,SUBGRID,1))
  if classification == False:
    Y_all = np.ndarray(((n_bins**3)*4, SUBGRID,SUBGRID,SUBGRID,1),dtype=np.float16)
  else:
    Y_all = np.ndarray(((n_bins**3)*4, SUBGRID,SUBGRID,SUBGRID,1),dtype=np.int8)

  for i in range(n_bins):
    for j in range(n_bins):
      for k in range(n_bins):
        sub_den = den[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        X_all[cont,:,:,:,0] = sub_den
        sub_msk = msk[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        Y_all[cont,:,:,:,0] = sub_msk
        cont = cont+1

        sub_den = den[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        sub_den = volumes.rotate_cube(sub_den,2)
        X_all[cont,:,:,:,0] = sub_den
        sub_msk = msk[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        sub_msk = volumes.rotate_cube(sub_msk,2)
        Y_all[cont,:,:,:,0] = sub_msk
        cont = cont+1

        sub_den = den[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        sub_den = volumes.rotate_cube(sub_den,1)
        X_all[cont,:,:,:,0] = sub_den
        sub_msk = msk[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        sub_msk = volumes.rotate_cube(sub_msk,1)
        Y_all[cont,:,:,:,0] = sub_msk
        cont = cont+1

        sub_den = den[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        sub_den = volumes.rotate_cube(sub_den,0)
        X_all[cont,:,:,:,0] = sub_den
        sub_msk = msk[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        sub_msk = volumes.rotate_cube(sub_msk,0)
        Y_all[cont,:,:,:,0] = sub_msk
        cont = cont+1
    #print(i,j,k)
  X_all = X_all.astype('float32')
  Y_all = Y_all.astype('int8')
  gc.collect()
  return X_all, Y_all
#---------------------------------------------------------
# 4/30/24: adding load_dataset_all_overlap function
# this function will ACTUALLY do what we claim
# load_dataset has done all along, which is taking overlapping
# subcubes, rotating by 90 degrees 3 times, for 
# a total of N_subcubes = 4 * [(GRID/SUBGRID) + (GRID/SUBGRID - 1)]^3
# NOTE THAT THIS IS FOR TRAINING ONLY!!!
#---------------------------------------------------------
def load_dataset_all_overlap(FILE_DEN, FILE_MSK, SUBGRID, OFF, preproc='mm', sigma=None):
  '''
  Function that loads density and mask files, splits into overlapping subcubes.
  Subcubes overlap by OFF, and are of size SUBGRID.
  Each subcube is rotated by 90 deg three times.
  FILE_DEN: str filepath to density field.
  FILE_MSK: str filepath to mask field.
  SUBGRID: int size of subcubes.
  OFF: int overlap of subcubes.
  preproc: str preprocessing method. 'mm' for minmax, 'std' for standardize.
  sigma: float sigma for Gaussian smoothing. def None.
  '''
  print(f'Reading volume: {FILE_DEN}... ')
  den = volumes.read_fvolume(FILE_DEN)
  if sigma is not None:
    den = ndi.gaussian_filter(den,sigma,mode='wrap')
    print(f'Smoothed density with a Gaussian kernel of size {sigma}')
  print(f'Reading mask: {FILE_MSK}...')
  msk = volumes.read_fvolume(FILE_MSK)
  # print mask populations:
  _, cnts = np.unique(msk,return_counts=True)
  for val in cnts:
    print(f'% of population: {val/den.shape[0]**3 * 100}')
  summary(den); summary(msk)
  if preproc == 'mm':
    den = minmax(den)
    print('Ran preprocessing to scale density to [0,1]!')
    print('\nNew summary statistics for density field: ')
    summary(den)
  if preproc == 'std':
    den = standardize(den)
    print('Ran preprocessing by dividing density/mask by std dev and subtracting by the mean! ')
    print('\nNew summary statistics for density field: ')
    summary(den)
  nbins = den.shape[0]//SUBGRID + (den.shape[0]//SUBGRID - 1)
  print(f'Number of overlapping subcubes: {4*nbins**3}')
  X_all_overlap = np.ndarray(((nbins**3)*4, SUBGRID, SUBGRID, SUBGRID, 1))
  Y_all_overlap = np.ndarray(((nbins**3)*4, SUBGRID, SUBGRID, SUBGRID, 1))
  # loop over overlapping subcubes, rotate!
  cont = 0
  for i in range(nbins):
    off_i = SUBGRID*i - OFF*i
    for j in range(nbins):
      off_j = SUBGRID*j - OFF*j
      for k in range(nbins):
        off_k = SUBGRID*k - OFF*k
        # define subcube:
        sub_den = den[off_i:off_i+SUBGRID,off_j:off_j+SUBGRID,off_k:off_k+SUBGRID]
        sub_msk = msk[off_i:off_i+SUBGRID,off_j:off_j+SUBGRID,off_k:off_k+SUBGRID]
        X_all_overlap[cont,:,:,:,0] = sub_den
        Y_all_overlap[cont,:,:,:,0] = sub_msk
        cont += 1
        # rot 90:
        sub_den = np.rot90(sub_den)
        sub_msk = np.rot90(sub_msk)
        X_all_overlap[cont,:,:,:,0] = sub_den
        Y_all_overlap[cont,:,:,:,0] = sub_msk
        cont += 1
        # rot 180
        sub_den = np.rot90(sub_den)
        sub_msk = np.rot90(sub_msk)
        X_all_overlap[cont,:,:,:,0] = sub_den
        Y_all_overlap[cont,:,:,:,0] = sub_msk
        cont += 1
        # rot 270
        sub_den = np.rot90(sub_den)
        sub_msk = np.rot90(sub_msk)
        X_all_overlap[cont,:,:,:,0] = sub_den
        Y_all_overlap[cont,:,:,:,0] = sub_msk
        cont += 1
  gc.collect()
  return X_all_overlap.astype('float32'), Y_all_overlap.astype('int8')

#---------------------------------------------------------
# For loading testing/validation data for prediction
#---------------------------------------------------------
def load_dataset(file_in, SUBGRID, OFF, preproc='mm',sigma=None,return_int=False):
  #--- Read density field
  den = volumes.read_fvolume(file_in)
  if sigma is not None:
    den = ndi.gaussian_filter(den,sigma,mode='wrap')
    print(f'Density was smoothed w/ a Gaussian kernel of size {sigma}')
  if preproc == 'mm':
    #den = minmax(np.log10(den)) # MUST MATCH PREPROC METHOD USED IN TRAIN
    den = minmax(den); print('Ran preprocessing to scale density to [0,1]!')
  if preproc == 'std':
    den = standardize(den); print('Ran preprocessing to scale density s.t. mean=0 and std dev = 1!')
  if preproc == None:
    pass
  #nbins = (den.shape[0] // SUBGRID) + 1 + 1 + 1 # hacky way
  #if den.shape[0] == 640:
  #  nbins += 1
  nbins = den.shape[0]//SUBGRID + (den.shape[0]//SUBGRID - 1)
  X_all = np.zeros(shape=(nbins**3, SUBGRID,SUBGRID,SUBGRID,1))
  
  cont  = 0
  for i in range(nbins):
    off_i = SUBGRID*i - OFF*i
    for j in range(nbins):
      off_j = SUBGRID*j - OFF*j
      for k in range(nbins):
        off_k = SUBGRID*k - OFF*k
        #print(i,j,k,'|', off_i,':',off_i+SUBGRID,',',off_j,':',off_j+SUBGRID,',',off_k,':',off_k+SUBGRID)
        sub_den = den[off_i:off_i+SUBGRID,off_j:off_j+SUBGRID,off_k:off_k+SUBGRID]
        X_all[cont,:,:,:,0] = sub_den
        cont = cont+1
      
  if return_int:
    X_all = X_all.astype('uint8')
  else:
    X_all = X_all.astype('float16')
  gc.collect()
  return X_all
#---------------------------------------------------------
# Focal loss function
#---------------------------------------------------------
#@register_keras_serializable()
def categorical_focal_loss(alpha, gamma=2.):
  """
  Softmax version of focal loss.
  When there is a skew between different categories/labels in your data set, you can try to apply this function as a
  loss.
          m
    FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
        c=1

    where m = number of classes, c = class and o = observation

  Parameters:
    alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
    categories/labels, the size of the array needs to be consistent with the number of classes.
    gamma -- focusing parameter for modulating factor (1-p)

  Default value:
    gamma -- 2.0 as mentioned in the paper
    alpha -- 0.25 as mentioned in the paper

  References:
      Official paper: https://arxiv.org/pdf/1708.02002.pdf
      https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

  Usage:
    model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
  """

  alpha = np.array(alpha, dtype=np.float32)
  #@register_keras_serializable()
  def categorical_focal_loss_fixed(y_true, y_pred):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred: A tensor resulting from a softmax
    :return: Output tensor.
    """
    # if the dtype of y_true is uint8, cast it to float32
    if y_true.dtype == 'uint8':
      y_true = tf.cast(y_true, 'float32')
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    # Calculate Cross Entropy
    cross_entropy = -y_true * K.log(y_pred)

    # Calculate Focal Loss
    loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

    # Compute mean loss in mini_batch
    return K.mean(K.sum(loss, axis=-1))

  return categorical_focal_loss_fixed
#---------------------------------------------------------
# Custom combo SCCE and Dice loss function
#---------------------------------------------------------
def SCCE_Dice_loss(y_true, y_pred, cce_weight=1.0, dice_weight=1.0):
  """
  DISCCE loss
  Custom loss function combining sparse categorical cross-entropy (SCCE) and dice score.
  This uses the macro-averaged Dice score. NOTE could be modified to use micro-averaged 
  dice score, or some kind of weighted averaging.

  Args:
    y_true: True labels.
    y_pred: Predicted labels.
    cce_weight: Weight for SCCE loss (default: 1.0).
    dice_weight: Weight for dice score loss (default: 1.0).

  Returns:
    Weighted average of SCCE loss and dice score.
  """
  # cast y_true to float32
  y_true = tf.cast(y_true, tf.float32)
  # Calculate SCCE loss
  scce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

  # Calculate multi-class dice score
  smooth = 1e-5
  intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
  union = tf.reduce_sum(y_true, axis=-1) + tf.reduce_sum(y_pred, axis=-1)
  dice_score = (2 * intersection + smooth) / (union + smooth)
  dice_score = tf.reduce_mean(dice_score)

  # Weighted average of SCCE loss and dice score
  return cce_weight * scce_loss + dice_weight * (1 - dice_score)
#---------------------------------------------------------
# U-Net creator
#---------------------------------------------------------
def conv_block(input_tensor, filters, name, activation='relu', batch_normalization=True, dropout_rate=None, BN_scheme='last', DROP_scheme='last', kernel_regularizer=None, kernel_initializer='he_normal'):
  '''
  Convolutional block for U-Net. 
  input_tensor: input tensor
  filters: int number of filters
  name: str name of block
  activation: str activation function. Default is 'relu'. Options are 'relu', 'elu', 'selu', 'tanh', 'sigmoid', 'softmax', 'LeakyReLU'
  batch_normalization: bool whether or not to use batch normalization
  dropout_rate: float dropout rate
  BN_scheme: str batch normalization scheme. 'last' = last conv layer of block, 'all' = both layers, 'none' = no batch normalization.
  DROP_scheme: str dropout scheme. 'last' = last conv layer of block, 'all' = both layers, 'none' = no dropout. needs to be none if dropout_rate is 0.0
  kernel_regularizer: float regularizer for kernel weights. Default is None. Only L2 regularization is supported.
  kernel_initializer: str initializer for kernel weights. Default is 'he_normal'. Options are 'he_normal', 'he_uniform', 'glorot_normal', 'glorot_uniform', 'lecun_normal', 'lecun_uniform'.

  here we assume that BN layers should come after the conv layers and before the activation layers.
  we assume that drop layers should come after the activation layers.
  '''
  if dropout_rate == 0.0:
    DROP_scheme = 'none'
  if batch_normalization == False:
    BN_scheme = 'none'
  if kernel_regularizer is not None:
    kernel_regularizer = regularizers.l2(kernel_regularizer)
  
  x = Conv3D(filters, kernel_size=(3, 3, 3), padding='same',
             name=name, kernel_regularizer=kernel_regularizer,
             kernel_initializer=kernel_initializer)(input_tensor)
  if BN_scheme == 'all':
    x = BatchNormalization()(x)
  x = Activation(activation)(x)
  if DROP_scheme == 'all':
    x = Dropout(dropout_rate)(x)
  x = Conv3D(filters, kernel_size=(3, 3, 3), padding='same', 
             name=name+'_1', kernel_regularizer=kernel_regularizer,
             kernel_initializer=kernel_initializer)(x)
  if BN_scheme == 'all' or BN_scheme == 'last':
    x = BatchNormalization()(x)
  x = Activation(activation)(x)
  if DROP_scheme == 'all' or DROP_scheme == 'last':
    x = Dropout(dropout_rate/2.0)(x) # only half the dropout rate here????
  return x
#---------------------------------------------------------
def unet_3d(input_shape, num_classes=4, initial_filters=16, depth=4, activation='relu', last_activation='softmax', batch_normalization=False, BN_scheme='last', dropout_rate=None, DROP_scheme='last', REG_FLAG = False, model_name='3D_U_Net', report_params=False):
  '''
  Constructs a 3D U-Net model for semantic segmentation.

  Parameters:
  - input_shape (tuple): The shape of the input tensor (excluding batch dimension). i.e. (None, None, None, 1)
  - num_classes (int): The number of output classes. Default is 4 (void, wall, filament, halo).
  - initial_filters (int): The number of filters in the first convolutional layer. Default is 16.
  - depth (int): The depth of the U-Net architecture. Default is 4.
  - activation (str): The activation function to use in the convolutional layers. Default is 'relu'.
  - last_activation (str): The activation function to use in the output layer. Default is 'softmax'.
  - batch_normalization (bool): Whether to use batch normalization after each convolutional layer. Default is False.
  - BN_scheme (str): The batch normalization scheme to use. 'all' = after each conv layer, 'last' = after last conv layer, 'none' = no batch normalization. Default is 'last'.
  - dropout_rate (float): The dropout rate to use after each convolutional layer. Default is None.
  - DROP_scheme (str): The dropout scheme to use. 'all' = after each conv layer, 'last' = after last conv layer, 'none' = no dropout. Default is 'last'.
  - REG_FLAG (bool): Whether to use L2 regularization on the kernel weights. Default is False.
  - model_name (str): The name of the model. Default is '3D_U_Net'.
  - report_params (bool): Whether to return the number of trainable and non-trainable parameters. Default is False.

  Returns:
  - model (tf.keras.Model): The constructed 3D U-Net model.
  - trainable_ps (int): The number of trainable parameters in the model (if report_params=True).
  - nontrainable_ps (int): The number of non-trainable parameters in the model (if report_params=True).
  '''
  # check if dropout rate is 0.0
  if dropout_rate == 0.0 or dropout_rate == None:
    DROP_scheme = 'none'
  # check if batch normalization is False
  if batch_normalization == False:
    BN_scheme = 'none'
  # check if L2 regularization is used
  if REG_FLAG == False:
    REG_FLAG = None
  # Input
  inputs = Input(input_shape)
  
  # Encoder path
  encoder_outputs = []
  x = inputs
  for d in range(depth):
    filters = initial_filters * (2 ** d)
    block_name = f'encoder_block_D{d}'
    x = conv_block(x, filters, block_name, activation, batch_normalization,
                   dropout_rate, BN_scheme, DROP_scheme, REG_FLAG)
    encoder_outputs.append(x)
    x = MaxPooling3D(pool_size=(2, 2, 2),name=block_name+'_maxpool')(x)
  
  # Bottom
  x = conv_block(x, initial_filters * (2 ** depth), 'bottleneck', activation,
                 batch_normalization, dropout_rate, BN_scheme, DROP_scheme,
                 REG_FLAG)
  
  # Decoder path
  for d in reversed(range(depth)):
    filters = initial_filters * (2 ** d)
    block_name = f'decoder_block_D{d}'
    x = UpSampling3D(size=(2, 2, 2),name=block_name+'_upsample')(x)
    x = Concatenate(axis=-1,name=block_name+'_concat')([x, encoder_outputs[d]])
    x = conv_block(x, filters, block_name, activation,
                   batch_normalization, dropout_rate,
                   BN_scheme, DROP_scheme, REG_FLAG)
  
  # Output
  outputs = Conv3D(num_classes, kernel_size=(1, 1, 1), name='output_conv')(x)
  if last_activation is not None:
    outputs = Activation(last_activation)(outputs)
  
  model = Model(inputs=inputs, outputs=outputs, name=model_name)
  # calculate number of parameters:
  trainable_ps = layer_utils.count_params(model.trainable_weights)
  nontrainable_ps = layer_utils.count_params(model.non_trainable_weights)
  print(f'Total params: {trainable_ps + nontrainable_ps}')
  print(f'Trainable params: {trainable_ps}')
  print(f'Non-trainable params: {nontrainable_ps}')
  if report_params == True:
    return model, trainable_ps, nontrainable_ps
  return model
#---------------------------------------------------------
# Saving model parameters to a txt file
#---------------------------------------------------------
def save_dict_to_text(dictionary, file_path):
  with open(file_path, 'w') as file:
    for key, value in dictionary.items():
      file.write(f'{key}: {value}\n')
#---------------------------------------------------------
# Better, faster, tf native metric calculations:
# Functions for F1, precision, recall, MCC, etc. written
# using keras.backend functions.
#---------------------------------------------------------
def PR_F1_keras(int_labels=True):
  def PR_F1_macro(y_true, y_pred):
    '''
    Precision, Recall, F1 score metrics using keras.backend functions.
    NOTE that this is faster than individually calcing prec, recall,
    F1 score.
    y_true: true labels
    y_pred: predicted labels
    int_labels: bool whether or not labels are integer or one-hot. def True.
    Returns: tuple of precision, recall, F1 score. (macro-avg)
    '''
    if not int_labels:
      y_true = K.argmax(y_true, axis=-1)
      y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    TP = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    FP = K.sum(K.cast((1-y_true) * y_pred, 'float'), axis=0)
    FN = K.sum(K.cast(y_true * (1-y_pred), 'float'), axis=0)
    precision = TP / (TP + FP + K.epsilon())
    recall = TP / (TP + FN + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return precision, recall, f1
  return PR_F1_macro
def PR_F1_micro_keras(int_labels=True):
  def PR_F1_micro(y_true, y_pred):
    '''
    Precision, Recall, F1 score metrics using keras.backend functions.
    NOTE that this is faster than individually calcing prec, recall,
    F1 score.
    y_true: true labels
    y_pred: predicted labels
    int_labels: bool whether or not labels are integer or one-hot. def True.
    Returns: tuple of precision, recall, F1 score. (micro-avg)
    '''
    if not int_labels:
      y_true = K.argmax(y_true, axis=-1)
      y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    TP = K.sum(K.cast(y_true * y_pred, 'float'))
    FP = K.sum(K.cast((1-y_true) * y_pred, 'float'))
    FN = K.sum(K.cast(y_true * (1-y_pred), 'float'))
    precision = TP / (TP + FP + K.epsilon())
    recall = TP / (TP + FN + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return precision, recall, f1
  return PR_F1_micro
def precision_keras(num_classes=4, int_labels=True):
  def precision_macro(y_true, y_pred):
    '''
    Compute macro precision using keras.backend functions.
    y_true: true labels.
    y_pred: predicted labels.
    num_classes: int number of classes. def 4.
    int_labels: bool whether or not labels are integer or one-hot. def True.
    Returns: precision. tf.Tensor.
    '''
    if not int_labels:
      y_true = K.argmax(y_true, axis=-1)
      y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    TP = K.sum(y_true * y_pred, axis=0)
    FP = K.sum((1-y_true) * y_pred, axis=0)
    precision = K.mean(TP / (TP + FP + K.epsilon()))
    return precision
  return precision_macro
def recall_keras(num_classes=4, int_labels=True):
  def recall_macro(y_true, y_pred):
    '''
    Compute macro recall using keras.backend functions.
    y_true: true labels.
    y_pred: predicted labels.
    num_classes: int number of classes. def 4.
    int_labels: bool whether or not labels are integer or one-hot. def True.
    Returns: recall. tf.Tensor.
    '''
    if not int_labels:
      y_true = K.argmax(y_true, axis=-1)
      y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    TP = K.sum(y_true * y_pred, axis=0)
    FN = K.sum(y_true * (1-y_pred), axis=0)
    recall = K.mean(TP / (TP + FN + K.epsilon()))
    return recall
  return recall_macro
def F1_keras(num_classes=4, int_labels=True):
  def F1_macro(y_true, y_pred):
    '''
    Compute macro F1 score using keras.backend functions.
    y_true: true labels.
    y_pred: predicted labels.
    num_classes: int number of classes. def 4.
    int_labels: bool whether or not labels are integer or one-hot. def True.
    Returns: F1 score. tf.Tensor.
    '''
    precision = precision_keras(num_classes, int_labels)(y_true, y_pred)
    recall = recall_keras(num_classes, int_labels)(y_true, y_pred)
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return f1
  return F1_macro
def precision_micro_keras(num_classes=4, int_labels=True):
  def precision_micro(y_true, y_pred):
    '''
    Compute micro precision using keras.backend functions.
    y_true: true labels.
    y_pred: predicted labels.
    num_classes: int number of classes. def 4.
    int_labels: bool whether or not labels are integer or one-hot. def True.
    Returns: micro precision. tf.Tensor.
    '''
    if not int_labels:
      y_true = K.argmax(y_true, axis=-1)
      y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    TP = K.sum(y_true * y_pred)
    FP = K.sum((1-y_true) * y_pred)
    precision = TP / (TP + FP + K.epsilon())
    return precision
  return precision_micro
def recall_micro_keras(num_classes=4, int_labels=True):
  def recall_micro(y_true, y_pred):
    '''
    Compute micro recall using keras.backend functions.
    y_true: true labels.
    y_pred: predicted labels.
    num_classes: int number of classes. def 4.
    int_labels: bool whether or not labels are integer or one-hot. def True.
    Returns: micro recall. tf.Tensor.
    '''
    if not int_labels:
      y_true = K.argmax(y_true, axis=-1)
      y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    TP = K.sum(y_true * y_pred)
    FN = K.sum(y_true * (1-y_pred))
    recall = TP / (TP + FN + K.epsilon())
    return recall
  return recall_micro
def F1_micro_keras(num_classes=4, int_labels=True):
  def F1_micro(y_true, y_pred):
    '''
    Compute micro F1 score using keras.backend functions.
    y_true: true labels.
    y_pred: predicted labels.
    num_classes: int number of classes. def 4.
    int_labels: bool whether or not labels are integer or one-hot. def True.
    Returns: micro F1 score. tf.Tensor.
    '''
    precision = precision_micro_keras(num_classes, int_labels)(y_true, y_pred)
    recall = recall_micro_keras(num_classes, int_labels)(y_true, y_pred)
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return f1
  return F1_micro
def MCC_keras(num_classes=4, int_labels=True):
  def MCC(y_true, y_pred):
    '''
    Matthews correlation coefficient using keras.backend functions.
    y_true: true labels (shape: [N_samples, SUBGRID, SUBGRID, SUBGRID, 1])
    y_pred: predicted labels (if int_labels=True, shape: [N_samples, SUBGRID, SUBGRID, SUBGRID, 1])
    (if int_labels=False, shape: [N_samples, SUBGRID, SUBGRID, SUBGRID, num_classes])
    num_classes: int number of classes. def 4.
    int_labels: bool whether or not labels are integer or one-hot. def True.
    Returns: Matthews correlation coefficient.
    '''
    y_pred = K.cast(K.argmax(y_pred, axis=-1), 'int32')
    # if last shape is 4, argmax it.
    if y_true.shape[-1] == 4:
      y_true = K.cast(K.argmax(y_true, axis=-1), 'int32')
    else:
      y_true = K.cast(K.squeeze(y_true, axis=-1), 'int32')
    #if not int_labels:
    #  y_true = K.cast(K.argmax(y_true, axis=-1), 'int32')
    #else:
    #  y_true = K.cast(K.squeeze(y_true, axis=-1), 'int32')
    # reshape y_true and y_pred to 1D tensors.
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    # calculate confusion matrix.
    C = tf.math.confusion_matrix(
      labels=y_true,
      predictions=y_pred,
      num_classes=num_classes,
      dtype=tf.int32,
      weights=None,
    )
    # cast confusion matrix to float32.
    t_sum = K.cast(tf.reduce_sum(C,axis=1), 'float32')
    p_sum = K.cast(tf.reduce_sum(C,axis=0), 'float32')
    n_correct = K.cast(tf.linalg.trace(C), 'float32')
    n_samples = K.cast(tf.reduce_sum(p_sum), 'float32')
    # calculate MCC.
    cov_ytyp = n_correct * n_samples - tf.tensordot(t_sum, p_sum, axes=1)
    cov_ypyp = n_samples**2 - tf.tensordot(p_sum, p_sum, axes=1)
    cov_ytyt = n_samples**2 - tf.tensordot(t_sum, t_sum, axes=1)
    mcc_value = cov_ytyp / K.sqrt(cov_ytyt * cov_ypyp + K.epsilon())
    # handle NaN:
    mcc_value = K.switch(tf.math.is_nan(mcc_value), K.zeros_like(mcc_value), mcc_value)
    return mcc_value
  return MCC
def balanced_accuracy_keras(num_classes=4, int_labels=True):
  def balanced_accuracy(y_true, y_pred):
    '''
    Balanced accuracy using keras.backend functions.
    y_true: true labels
    y_pred: predicted labels
    num_classes: int number of classes. def 4.
    int_labels: bool whether or not labels are integer or one-hot. def True.
    Returns: balanced accuracy.
    '''
    if not int_labels:
      y_true = K.argmax(y_true, axis=-1)
      y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    TP = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    FN = K.sum(K.cast(y_true * (1-y_pred), 'float'), axis=0)
    recall_per_class = TP / (TP + FN + K.epsilon())
    balanced_accuracy = K.mean(recall_per_class)
    return balanced_accuracy
  return balanced_accuracy
def void_PR_F1_keras(num_classes=4, int_labels=True):
  def void_PR_F1(y_true, y_pred):
    '''
    Precision, Recall, F1 score for the void class [0] using keras
    backend functions.
    y_true: true labels
    y_pred: predicted labels
    num_classes: int number of classes. def 4.
    int_labels: bool whether or not labels are integer or one-hot. def True.
    Returns: tuple of precision, recall, F1 score for the void class.
    '''
    if not int_labels:
      y_true = K.argmax(y_true, axis=-1)
      y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)
    void_true = K.cast(K.equal(y_true, 0), 'float')
    void_pred = K.cast(K.equal(y_pred, 0), 'float')
    TP = K.sum(void_true * void_pred)
    FP = K.sum((1-void_true) * void_pred)
    FN = K.sum(void_true * (1-void_pred))
    precision = TP / (TP + FP + K.epsilon())
    recall = TP / (TP + FN + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return precision, recall, f1
  return void_PR_F1
def void_F1_keras(num_classes=4, int_labels=True):
  def void_F1(y_true, y_pred):
    '''
    F1 score for the void class [0] using keras backend functions.
    y_true: true labels
    y_pred: predicted labels
    num_classes: int number of classes. def 4.
    int_labels: bool whether or not labels are integer or one-hot. def True.
    Returns: F1 score for the void class.
    '''
    if not int_labels:
      y_true = K.argmax(y_true, axis=-1)
      y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)
    void_true = K.cast(K.equal(y_true, 0), 'float')
    void_pred = K.cast(K.equal(y_pred, 0), 'float')
    TP = K.sum(void_true * void_pred)
    FP = K.sum((1-void_true) * void_pred)
    FN = K.sum(void_true * (1-void_pred))
    precision = TP / (TP + FP + K.epsilon())
    recall = TP / (TP + FN + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return f1
  return void_F1
def true_wall_pred_as_void_keras(num_classes=4, int_labels=True):
  def true_wall_pred_as_void(y_true, y_pred):
    '''
    Calculates the number of true wall voxels predicted as void normalized by
    the total number of wall voxels using keras.backend functions.
    y_true: true labels
    y_pred: predicted labels
    num_classes: int number of classes. def 4.
    int_labels: bool whether or not labels are integer or one-hot. def True.
    Returns: true wall predicted as void.
    '''
    if not int_labels:
      y_true = K.argmax(y_true, axis=-1)
      y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)
    wall_true = K.cast(K.equal(y_true, 1), 'float')
    void_pred = K.cast(K.equal(y_pred, 0), 'float')
    true_wall_pred_as_void = K.sum(wall_true * (1-void_pred) * void_pred) / (K.sum(wall_true) + K.epsilon())
    return true_wall_pred_as_void
  return true_wall_pred_as_void
#---------------------------------------------------------
# Custom metric class for computing metrics using keras
# backend functions.
#---------------------------------------------------------
all_TF_metrics = ['macro_F1','macro_precision','macro_recall','micro_F1','micro_precision',
                  'micro_recall','balanced_accuracy','matt_corrcoef','void_F1','void_precision',
                  'void_recall','true_wall_pred_as_void']
required_TF_metrics = ['macro_F1','macro_precision','macro_recall','matt_corrcoef']
class keras_ComputeMetrics(metrics.Metric):
  '''
  Class containing way to compute metrics using TensorFlow native functions.
  Hopefully this is faster than the old ComputeMetrics class which relied
  on sklearn functions. 
  NOTE that this is a metric, and as such only computes stats that can 
  be calculated a batch at a time. Note that the old ComputeMetrics callback
  calculated metrics across the whole dataset, like ROC AUC.

  This metric will calculate (some optionally, can specify which to calc):
  - F1 score (req)
  - Precision (req)
  - Recall (req)
  - Matthews correlation coefficient (req)
  - Balanced accuracy
  - Void F1
  - Void Precision
  - Void Recall
  - True Wall predicted as Void, normalized by total Wall (as in CM figure).
  '''
  def __init__(self, num_classes=4, name='ComputeMetrics', metrics=required_TF_metrics, **kwargs):
    super(keras_ComputeMetrics, self).__init__(name=name, **kwargs)
    self.num_classes = num_classes
    self.metrics = metrics
    self.tp = self.add_weight(name='true_positives', initializer='zeros')
    self.fp = self.add_weight(name='false_positives', initializer='zeros')
    self.tn = self.add_weight(name='true_negatives', initializer='zeros')
    self.fn = self.add_weight(name='false_negatives', initializer='zeros')
  def update_state(self, y_true, y_pred, sample_weight=None):
    pass

#---------------------------------------------------------
# Callback to calculate more classification metrics:
#---------------------------------------------------------
class ComputeMetrics(Callback):
  '''
  This metric computes the F1 score, precision, recall, balanced accuracy,
  Matthews correlation coefficient, and ROC AUC for a multi-class classification.
  It also computes the F1 score, recall, and precision for the void class.
  Usage: metrics = ComputeMetrics((X_test,Y_test),10,avg='macro')
  avg can be: 'micro', 'macro', 'weighted'.
  N_epochs is how often you want to compute the metrics. 
  beta is for the Fscore (default=1, F1 score = Dice coef)
  one_hot: bool, added so that SCCE models don't error out during training.
  '''
  def __init__(self,val_data,N_epochs,avg='micro',beta=1.0,one_hot=True):
    super().__init__()
    self.validation_data = val_data
    self.N_epochs = N_epochs
    self.avg = avg
    self.beta = beta
    self.one_hot = one_hot
  def on_epoch_end(self,epoch,logs={}):
    if epoch % self.N_epochs == 0:
      X_test = self.validation_data[0]; Y_test = self.validation_data[1]
      Y_pred = self.model.predict(X_test,verbose=0) # last axis has 4 channels
      #_val_loss, _val_acc = self.model.evaluate(X_test,Y_test,verbose=0)
      # ROC_AUC needs to be ran on one-hot encoded data
      if not self.one_hot:
        Y_test = to_categorical(Y_test,num_classes=4)
      _val_ROC_AUC = roc_auc_score(Y_test.reshape(-1,4),Y_pred.reshape(-1,4),average=self.avg,multi_class='ovr')
      Y_test = np.argmax(Y_test,axis=-1); Y_test = np.expand_dims(Y_test,axis=-1)
      Y_pred = np.argmax(Y_pred,axis=-1); Y_pred = np.expand_dims(Y_pred,axis=-1)
      Y_test = Y_test.ravel(); Y_pred = Y_pred.ravel()
      _val_balanced_acc = balanced_accuracy_score(Y_test,Y_pred)
      _val_precision, _val_recall, _val_f1, _ = precision_recall_fscore_support(Y_test,Y_pred,beta=self.beta,average=self.avg,zero_division=0.0)
      _val_matt_corrcoef = matthews_corrcoef(Y_test,Y_pred)
      _val_void_precision, _val_void_recall, _val_void_f1, _ = precision_recall_fscore_support(Y_test,Y_pred,beta=self.beta,average=None,labels=[0],zero_division=0.0)

      #logs['val_loss'] = _val_loss
      #logs['val_acc'] = _val_acc
      logs['val_balanced_acc'] = _val_balanced_acc
      logs['val_f1'] = _val_f1
      logs['val_recall'] = _val_recall
      logs['val_precision'] = _val_precision
      logs['val_ROC_AUC'] = _val_ROC_AUC
      logs['val_matt_corrcoef'] = _val_matt_corrcoef
      logs['val_void_f1'] = _val_void_f1[0]
      logs['val_void_recall'] = _val_void_recall[0]
      logs['val_void_precision'] = _val_void_precision[0]
      gc.collect()

      #print(f' - Balanced Acc: {_val_balanced_acc:.4f} - F1: {_val_f1:.4f} - Precision: {_val_precision:.4f} - Recall: {_val_recall:.4f} - ROC AUC: {_val_ROC_AUC:.4f} \nMatt Corr Coef: {_val_matt_corrcoef:.4f} - Void F1: {_val_void_f1:.4f} - Void Recall: {_val_void_recall:.4f} - Void Precision: {_val_void_precision:.4f}')
      return
    else:
      #logs['val_loss'] = np.nan
      #logs['val_acc'] = np.nan
      logs['val_balanced_acc'] = np.nan
      logs['val_f1'] = np.nan
      logs['val_recall'] = np.nan
      logs['val_precision'] = np.nan
      logs['val_ROC_AUC'] = np.nan
      logs['val_matt_corrcoef'] = np.nan
      logs['val_void_f1'] = np.nan
      logs['val_void_recall'] = np.nan
      logs['val_void_precision'] = np.nan
      gc.collect()
#---------------------------------------------------------
# Scoring functions for multi-class classification
#---------------------------------------------------------
def F1s(y_true, y_pred, FILE_MODEL, score_dict):
  '''
  helper fxn for save_scores_from_fvol to calculate F1 scores
  and write to score_dict dictionary.
  NOTE while this function is called F1s, it really calcs:
  F1 score, recall, precision, and balanced accuracy.
  and matthew correlation coefficient!
  '''
  #FILE_HPTXT = FILE_MODEL + '_hps.txt'
  MODEL_NAME = FILE_MODEL.split('/')[-1]
  # calculate F1 scores:
  ps, rs, f1s, _ = precision_recall_fscore_support(y_true.ravel(), y_pred.ravel(), average=None,zero_division=0.0)
  micro_f1 = f1_score(y_true.ravel(), y_pred.ravel(), average='micro')
  macro_f1 = f1_score(y_true.ravel(), y_pred.ravel(), average='macro')
  weight_f1 = f1_score(y_true.ravel(), y_pred.ravel(), average='weighted')
  bal_acc = balanced_accuracy_score(y_true.ravel(),y_pred.ravel())
  mcc = matthews_corrcoef(y_true.ravel(),y_pred.ravel())
  # NOTE WRITING SCORES TO HYPERPARAMETER TXT FILES IS DEPRECATED!
  #with open(FILE_HPTXT, 'a') as f:
  #  for i in range(len(f1s)):
  #    f.write(f'Class {class_labels[i]} F1: {f1s[i]} \n')
  #  f.write(f'\nAverage F1: {np.mean(f1s)} \n')
  # add to score_dict:
  score_dict['micro_f1'] = micro_f1
  score_dict['macro_f1'] = macro_f1
  score_dict['weighted_f1'] = weight_f1
  score_dict['balanced_accuracy'] = bal_acc
  score_dict['matt_corrcoef'] = mcc
  for i in range(len(f1s)):
    score_dict[f'class_{class_labels[i]}_f1'] = f1s[i]
    score_dict[f'class_{class_labels[i]}_precision'] = ps[i]
    score_dict[f'class_{class_labels[i]}_recall'] = rs[i]
    print(f'Class {class_labels[i]} F1: {f1s[i]}')
    print(f'Class {class_labels[i]} precision: {ps[i]}')
    print(f'Class {class_labels[i]} recall: {rs[i]}')
  print(f'Micro F1: {micro_f1} \nMacro F1: {macro_f1} \nWeighted F1: {weight_f1}')
  print(f'Balanced accuracy: {bal_acc}')
  print(f'Matthews correlation coefficient: {mcc}')

def compute_metrics(y_true, y_pred):
  '''
  For parallelization purposes. This function will calculate
  the F1 score, precision, recall for a multi-class classification.
  (assumes y_true and y_pred are already raveled)
  '''
  ps, rs, f1s, _ = precision_recall_fscore_support(y_true, y_pred, average=None,zero_division=0.0)
  micro_f1 = f1_score(y_true, y_pred, average='micro')
  return ps, rs, f1s, micro_f1

def split_into_chunks(y_true, y_pred, chunk_size):
  '''
  Helper function for save_scores_from_fvol to split the
  true and predicted labels into chunks for parallel
  processing. Assumes y_true and y_pred are already raveled.
  '''
  n_samples = len(y_true)
  for i in range(0, n_samples, chunk_size):
    yield y_true[i:i+chunk_size], y_pred[i:i+chunk_size]

def parallel_compute_metrics(y_true, y_pred, chunk_size):
  chunks = list(split_into_chunks(y_true, y_pred, chunk_size))
  results = []
  with ThreadPoolExecutor() as executor:
    futures = [executor.submit(compute_metrics, chunk[0], chunk[1]) for chunk in chunks]
    for future in futures:
      results.append(future.result())
  return results

def CMatrix(y_true, y_pred, FILE_MODEL, FILE_FIG):
  '''
  helper fxn for save_scores_from_fvol to plot confusion matrix
  and save a text version in hyperparameters txt file.
  '''
  FILE_HPTXT = FILE_MODEL + '_hps.txt'
  MODEL_NAME = FILE_MODEL.split('/')[-1]
  # compute confusion matrix:
  plt.rcParams.update({'font.size': 14})
  cm = confusion_matrix(y_true.ravel(), y_pred.ravel(),
                        labels=[0,1,2,3],normalize='true')
  class_report = classification_report(y_true.ravel(), y_pred.ravel(),labels=[0,1,2,3],output_dict=True)
  # write confusion matrix to hyperparameters txt file:
  with open(FILE_HPTXT, 'a') as f:
    f.write('\nConfusion matrix: \n')
    f.write(str(cm))
    # write classification report to hyperparameters txt file:
    f.write('\nClassification report: \n')
    f.write(str(class_report))
  # plot confusion matrix:
  fig, ax = plt.subplots(1,1,figsize=(8,8))
  display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_labels)
  _ = display.plot(ax=ax)
  plt.savefig(FILE_FIG+MODEL_NAME+'_cm.png',facecolor='white',bbox_inches='tight')
  print(f'Saved confusion matrix to '+FILE_FIG+MODEL_NAME+'_cm.png')

def population_hists(y_true, y_pred, FILE_MODEL, FILE_FIG, FILE_DEN, n_bins=50):
  '''
  Helper function for save_scores_from_fvol to 
  write population percentages to hyperparameters txt
  file and plot histograms of the true and predicted
  masks and save in FILE_FIG.
  n_bins sets the number of bins in the log-log hist.

  NOTE- might want to add functionality for loading different 
  density files depending on the model. For now just load 
  full TNG300-3-Dark DM density.
  '''
  FILE_HPTXT = FILE_MODEL + '_hps.txt'
  MODEL_NAME = FILE_MODEL.split('/')[-1]
  y_true_vals, y_true_cts = np.unique(y_true, return_counts=True)
  y_pred_vals, y_pred_cts = np.unique(y_pred, return_counts=True)
  y_true_pcts = np.round(y_true_cts/np.sum(y_true_cts)*100,1)
  y_pred_pcts = np.round(y_pred_cts/np.sum(y_pred_cts)*100,1)
  with open(FILE_HPTXT, 'a') as f:
    f.write('\nPredicted population percentages: \n')
    for i in range(len(class_labels)):
      f.write(f'Class {class_labels[i]}: {y_pred_pcts[i]}% \n')
  # plot histograms:
  #d = volumes.read_fvolume('/ifs/groups/vogeleyGrp/data/TNG/DM_DEN_snap99_Nm=512.fvol')
  d = volumes.read_fvolume(FILE_DEN)
  d = d/np.mean(d) # we want delta+1
  plt.rcParams.update({'font.size': 16})
  fig, axs = plt.subplots(1,2,figsize=(12,8),tight_layout=True,sharey=True)
  axs[0].set_title('True Mask Population'); axs[1].set_title('Predicted Mask Population')
  bins = np.logspace(-5, 5, num=n_bins)
  for label in range(4):
    axs[0].hist(d[y_true==label], bins=bins, 
                label=class_labels[label]+' '+str(y_true_pcts[label])+'%', 
                alpha=0.5, histtype='step', log=True)
    axs[1].hist(d[y_pred==label], bins=bins, 
                label=class_labels[label]+' '+str(y_pred_pcts[label])+'%', 
                alpha=0.5, histtype='step', log=True)
  for ax in axs:
    ax.set_xlabel(r'$\delta$')
    ax.set_ylabel(r'$N(\delta)$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e1, None)
    ax.legend(loc='best',prop={'size': 11})
  plt.savefig(FILE_FIG+MODEL_NAME+'_hists.png',facecolor='white',bbox_inches='tight')
  print(f'Saved population hists of mask and pred to '+FILE_FIG+MODEL_NAME+'_hists.png')
     
def ROC_curves(y_true, y_pred, FILE_MODEL, FILE_FIG, score_dict, micro=True, macro=True, N_classes=4):
  '''
  Helper function for save_scores_from_fvol to plot ROC curves.
  # NOTE USE SOFTMAX PROBABILITY OUTPUTS FOR Y_PRED!!!!!
  # NOTE y_true, y_pred need to have be label binarized, aka shape=(N_samples, N_classes)
  y_true: true labels. shape: (N_samples, N_classes)
  y_pred: 4 channel probability outputs from softmax. shape: (N_samples, N_classes)
  FILE_MODEL: str, model filepath
  FILE_FIG: str, dir to save plot in
  score_dict: dict to store scores in
  micro, macro: bools, whether to plot micro/macro avg ROC curve. AUC for micro
  and macro will be saved to hyperparameters text anyway
  '''
  #FILE_HPTXT = FILE_MODEL + '_hps.txt'
  MODEL_NAME = FILE_MODEL.split('/')[-1]
  # plot ROC curves:
  plt.rcParams.update({'font.size': 16})
  fig, ax = plt.subplots(1,1,figsize=(12,12))
  ax.axis('square')
  ax.set_title('Multiclass One vs. Rest ROC Curves')
  # plot chance level (AUC=0.5)
  ax.plot([0,1],[0,1],linestyle='--',lw=2,color='r',label='Chance level')
  for i in range(N_classes):
    _ = RocCurveDisplay.from_predictions(
      y_true[:,i],
      y_pred[:,i],
      name=f'{class_labels[i]} ROC curve',
      ax=ax
    )
    print(f'Calculated ROC curve for class {class_labels[i]}!')
  # add micro averaged ROC curve:
  if micro:
    _ = RocCurveDisplay.from_predictions(
      y_true.ravel(),
      y_pred.ravel(),
      name='micro-avg ROC curve',
      ax=ax,
      linestyle=':'
    )
  if macro:
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(N_classes):
      fpr[i], tpr[i], _ = roc_curve(y_true[:,i],y_pred[:,i])
      roc_auc[i] = auc(fpr[i],tpr[i])
    fpr_grid = np.linspace(0.0,1.0,1000)
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(N_classes):
      mean_tpr += np.interp(fpr_grid,fpr[i],tpr[i])
    mean_tpr /= N_classes # avg over all classes
    fpr['macro'] = fpr_grid
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'],tpr['macro'])
    ax.plot(fpr['macro'],tpr['macro'],label=f"macro-avg ROC (AUC = {roc_auc['macro']:.2f})",linestyle=':')
  ax.legend(loc='best',prop={'size':11})
  ax.set_xlim(-0.05,1.05); ax.set_ylim(-0.05,1.05)
  _ = ax.set(
    xlabel = 'False Positive Rate',
    ylabel = 'True Positive Rate'
  )
  plt.savefig(FILE_FIG+MODEL_NAME+'_ROC_OvR.png',facecolor='white',bbox_inches='tight')
  print(f'Saved ROC OvR curves for each class to '+FILE_FIG+MODEL_NAME+'_ROC_OvR.png')
  # write AUCs to score_dict:
  micro_auc = roc_auc_score(y_true.ravel(),y_pred.ravel(),average='micro')
  class_aucs = roc_auc_score(y_true,y_pred,average=None,multi_class='ovr')
  score_dict['micro_ROC_AUC'] = micro_auc
  score_dict['macro_ROC_AUC'] = roc_auc['macro']
  for i in range(len(class_aucs)):
    score_dict[f'class_{class_labels[i]}_ROC_AUC'] = class_aucs[i]
  # NOTE WRITING SCORES TO HYPERPARAMETER TXT FILES IS DEPRECATED!
  #with open(FILE_HPTXT, 'a') as f:
  #  f.write(f'\nMicro-averaged ROC AUC: {micro_auc:.2f}\n')
  #  f.write(f'\nMacro-averaged ROC AUC: {roc_auc["macro"]:.2f}\n')
  #  for i in range(len(class_aucs)):
  #    f.write('\n'+f'Class {class_labels[i]} ROC AUC: {class_aucs[i]:.2f}\n')
  #print(f'Wrote ROC AUCs to '+FILE_HPTXT)

def PR_curves(y_true, y_pred, FILE_MODEL, FILE_FIG, score_dict, chance_lvl=False):
  '''
  function to plot Precision vs. Recall curves.
  # NOTE USE SOFTMAX PROBABILITY OUTPUTS FOR Y_PRED!!!!
  # NOTE y_true, y_pred need to have be label binarized, aka shape=(N_samples, N_classes)
  y_true: true labels. shape: (N_samples, N_classes)
  y_pred: 4 channel probability outputs from softmax. shape: (N_samples, N_classes)
  FILE_MODEL: str, model filepath
  FILE_FIG: str, dir to save plot in
  score_dict: dictionary to save scores in
  chance_lvl: bool, whether to plot chance level
  '''
  #FILE_HPTXT = FILE_MODEL + '_hps.txt'
  MODEL_NAME = FILE_MODEL.split('/')[-1]
  N_classes = len(class_labels)
  prec = dict(); recall = dict(); avg_prec = dict()
  # create fig, ax
  fig, ax = plt.subplots(1,1,figsize=(12,12))
  f_scores = np.linspace(0.3, 0.9, num=4)
  lines, labels = [], []
  for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    (l,) = ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    ax.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02),
                fontsize=10)
  for i in range(N_classes):
    prec[i], recall[i], _ = precision_recall_curve(y_true[:,i],y_pred[:,i])
    avg_prec[i] = average_precision_score(y_true[:,i],y_pred[:,i])
    print(f'Calculated PR curve for class {class_labels[i]}!')
  prec['micro'], recall['micro'], _ = precision_recall_curve(y_true.ravel(),y_pred.ravel())
  avg_prec['micro'] = average_precision_score(y_true,y_pred,average='micro')
  # plot micro avg PR curve:
  if chance_lvl:
    display = PrecisionRecallDisplay(
      recall=recall["micro"],
      precision=prec["micro"],
      average_precision=avg_prec["micro"],
      prevalence_pos_label=Counter(y_true.ravel())[1] / y_true.size
    )
    display.plot(ax=ax,name='Micro-avg PR',plot_chance_level=chance_lvl,linestyle=':')
  else:
    display = PrecisionRecallDisplay(
      recall=recall["micro"],
      precision=prec["micro"],
      average_precision=avg_prec["micro"]
    )
    display.plot(ax=ax,name='Micro-avg PR',linestyle=':')
  # plot each class PR curve:
  for i in range(N_classes):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=prec[i],
        average_precision=avg_prec[i],
    )
    display.plot(ax=ax,name=f'{class_labels[i]} PR')
  ax.set_xlim([0.0,1.0]); ax.set_ylim([0.0,1.05])
  handles, labels = ax.get_legend_handles_labels()
  handles.extend([l])
  labels.extend(["iso-F1 curves"])
  ax.legend(handles=handles, labels=labels, loc='best',prop={'size': 10})
  ax.set_title('Multi-Label Precision-Recall Curves')
  plt.savefig(FILE_FIG+MODEL_NAME+'_PR.png',facecolor='white',bbox_inches='tight')
  print('Saved precision-recall curves for each class at: '+FILE_FIG+MODEL_NAME+'_PR.png')
  # write results to score_dict:
  score_dict['micro_avg_AP'] = avg_prec['micro']
  for i in range(N_classes):
    score_dict[f'class_{class_labels[i]}_AP'] = avg_prec[i]
  # NOTE WRITING SCORES TO HYPERPARAMETER TXT FILES IS DEPRECATED!
  #with open(FILE_HPTXT, 'a') as f:
  #  f.write(f'\nMicro-averaged average precision: {avg_prec["micro"]:.2f}\n')
  #  for i in range(N_classes):
  #    f.write('\n'+f'Class {class_labels[i]} average precision: {avg_prec[i]:.2f}\n')
  #print(f'Wrote average precisions to '+FILE_HPTXT)

# 7/6/24: Create function to plot void voxels misclassified as wall voxels as 
# a certain color and vice versa. overlay on top of mask:
def plot_vw_misses(mask, pred, idx=None, Nm=512, boxsize=205., **kwargs):
  '''
  Plot and save a figure of void voxels misclassified as wall voxels and vice versa.
  mask: np.ndarray true mask
  pred: np.ndarray predicted mask
  idx: int index of slice to plot. def None, will just plot the middle slice.
  Nm: int number of voxels on a side in mask. def 512.
  boxsize: float size of box in Mpc/h. def 205.
  kwargs: dict of keyword arguments to pass to plt.imshow
  Returns: a matplotlib figure object.
  '''
  # get void voxels misclassified as wall voxels and vice versa:
  void_as_wall = (mask==0) & (pred==1)
  wall_as_void = (mask==1) & (pred==0)
  # make transparent for plotting on top of mask:
  void_as_wall = void_as_wall.astype(int); wall_as_void = wall_as_void.astype(int)
  void_as_wall = plotter.alpha0(void_as_wall); wall_as_void = plotter.alpha0(wall_as_void)
  # plot:
  if idx is None:
    idx = mask.shape[0]//2
  fig, ax = plt.subplots(1,1,figsize=(10,10))
  plotter.plot_arr(mask,idx,ax,segmented_cb=True,cmap='gray_r',**kwargs)
  plotter.plot_arr(void_as_wall,idx,ax,cb=False,cmap='Set1',**kwargs)
  plotter.plot_arr(wall_as_void,idx,ax,cb=False,cmap='tab10',**kwargs)
  title = f'Slice {idx} of Mask\nTrue void voxels misclassified as wall (red) \nTrue wall voxels misclassified as void (blue)'
  ax.set_title(title)
  # fix axis labels:
  plotter.set_window(0,boxsize,Nm,ax,boxsize)
  return fig

def save_scores_from_fvol(y_true, y_pred, FILE_MODEL, FILE_FIG, score_dict, N_CLASSES=4, VAL_FLAG=True, downsample=10):
  if not VAL_FLAG:
    print('WARNING: Model is being scored on training data. Scores may not be accurate.')
  if y_pred.shape[-1] != N_CLASSES:
    print(f'y_pred must be a {N_CLASSES} channel array of class probabilities. save_scores_from_fvol may not work as intended')

  # Binary classification adjustment
  if N_CLASSES == 2:
    print(f'Shape of y_pred before processing: {y_pred.shape}')
    print(f'Shape of y_true before processing: {y_true.shape}')
    # adjust single-channel preds to two-channel preds:
    #if y_pred.shape[-1] == 1:
      #y_pred = np.concatenate([1-y_pred, y_pred], axis=-1)
      #print(f'Adjusted y_pred shape: {y_pred.shape}')
    y_true_flat = y_true.flatten()
    #y_pred_flat = y_pred.reshape(-1, N_CLASSES)[:, 1]  # Select positive class probabilities
    y_pred_flat = y_pred.flatten()
    # Downsample
    y_true_flat = y_true_flat[::downsample]
    y_pred_flat = y_pred_flat[::downsample]
    # Handle invalid values
    valid_indices = ~np.isnan(y_pred_flat) & ~np.isnan(y_true_flat) & ~np.isinf(y_pred_flat)
    y_true_flat = y_true_flat[valid_indices]
    y_pred_flat = y_pred_flat[valid_indices]
    print(f'Processed y_true shape: {y_true_flat.shape}')
    print(f'Processed y_pred shape: {y_pred_flat.shape}')
    ROC_curves(y_true_flat, y_pred_flat, FILE_MODEL, FILE_FIG, score_dict,N_classes=N_CLASSES)
    PR_curves(y_true_flat, y_pred_flat, FILE_MODEL, FILE_FIG, score_dict)
    print('Saved ROC and PR curves for binary classification.')
  else:
    # Multi-class logic
    try:
      y_true_binarized = to_categorical(y_true, num_classes=N_CLASSES)
    except TypeError:
      y_true_binarized = to_categorical(y_true, num_classes=N_CLASSES)
    y_true_binarized = y_true_binarized.reshape(-1, N_CLASSES)
    y_pred_reshaped = y_pred.reshape(-1, N_CLASSES)
    # Downsample
    downsample_size = min(len(y_true_binarized), len(y_pred_reshaped))
    y_true_binarized = y_true_binarized[:downsample_size:downsample]
    y_pred_reshaped = y_pred_reshaped[:downsample_size:downsample]
    ROC_curves(y_true_binarized, y_pred_reshaped, FILE_MODEL, FILE_FIG, score_dict)
    PR_curves(y_true_binarized, y_pred_reshaped, FILE_MODEL, FILE_FIG, score_dict)
    print('Saved ROC and PR curves for multi-class classification.')

  # Argmax predictions and reshape for F1/Confusion Matrix
  y_pred = np.argmax(y_pred, axis=-1)
  y_pred = np.expand_dims(y_pred, axis=-1)
  F1s(y_true, y_pred, FILE_MODEL, score_dict)
  CMatrix(y_true, y_pred, FILE_MODEL, FILE_FIG)
  print('Saved metrics.')

'''
4/29/24: I'm tired of looking through model's hyperparameter txt files.
let's create a csv file that each training/prediction run will be appended to.
scores that we want to include: 
accuracy, loss, balanced acc, F1, precision, recall, ROC AUC, Matt Corr Coef, 
void f1, precision, recall, micro avg ROC AUC, macro avg ROC AUC, micro avg PR,
avg precisions for each class. 

also add VAL_FLAG to differentiate scores based on training data
from validation scores.

also I want to have fields for:
- SIMULATION trained on
- Depth
- Filters
- BN
- DROP
- UNIFORM_FLAG
- LOSS
- L the model was trained on 
- L the model was predicted on
(for training runs these are the same value.)
'''
def normalize_column_names(df):
  '''
  Normalize and strip whitespace from column names.
  Returns 
  '''
  regex = re.compile(r'(\.\d+\s*)+$')
  # Trim whitespace and remove sequences of .number with optional whitespace in between
  new_columns = [regex.sub('', col).strip() for col in df.columns]
  return new_columns
def consolidate_columns(df):
  '''
  Consolidate duplicated columns by keeping the first occurrence.
  '''
  # Normalize column names
  df.columns = normalize_column_names(df)
  # Consolidate duplicated columns by keeping the first occurrence
  df = df.loc[:, ~df.columns.duplicated(keep='first')]
  # Drop rows with all NaN values
  df = df.dropna(how='all')
  # strip whitespace from column names
  df = df.rename(columns=lambda x: x.strip())
  # strip whitespace from values
  df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
  return df
def save_scores_to_csv(score_dict, file_path):
  '''
  score_dict: dict of scores. 
  filepath: str of where to save scores.csv

  each dict of scores will be APPENDED to the csv, not overwritten.
  new cols will be added if they don't exist. 
  '''
  if os.path.isfile(file_path):
    print(f'>>> Loading model scores from {file_path}')
    df = pd.read_csv(file_path)
  else:
    print(f'>>> Creating new model scores file at {file_path}')
    df = pd.DataFrame()
  new_row = pd.DataFrame([score_dict])
  updated_df = pd.concat([df,new_row],ignore_index=True, sort=False)
  updated_df = consolidate_columns(updated_df)
  print(f'Removed duplicate columns and NaN rows.')
  updated_df.to_csv(file_path, index=False)
  print(f'>>> Appended scores to {file_path}')
col_names = [
  'SIM','L_TRAIN','L_PRED','DEPTH','FILTERS','BN','matt_corrcoef','balanced_accuracy',
  'micro_f1','weighted_f1','class_Void_f1','class_Wall_f1'
  ]
def save_scores_to_csv_small(score_dict, file_path, col_names=col_names):
  '''
  score_dict: dict of scores. 
  filepath: str of where to save scores.csv
  col_names: list of column names to save to csv. def col_names
  '''
  if os.path.isfile(file_path):
    print(f'>>> Loading model scores from {file_path}')
    df = pd.read_csv(file_path)
  else:
    print(f'>>> Creating new model scores file at {file_path}')
    df = pd.DataFrame(columns=col_names)
  new_row = pd.DataFrame([score_dict])
  updated_df = pd.concat([df,new_row],ignore_index=True, sort=False)
  updated_df = consolidate_columns(updated_df)
  print(f'Removed duplicate columns and NaN rows.')
  updated_df.to_csv(file_path, index=False)
  print(f'>>> Appended scores to {file_path}')
#---------------------------------------------------------
# Prediction functions:
#---------------------------------------------------------
def data_generator(data, batch_size):
  '''
  Generator function to feed data in batches to the model.
  Helps with OOM errors when predicting on a large volume.
  '''
  num_samples = data.shape[0]
  i = 0
  while True:
    batch_start = i * batch_size
    batch_end = (i + 1) * batch_size
    if batch_end >= num_samples:
      batch_end = num_samples

    yield data[batch_start:batch_end]
    i += 1
    if i * batch_size >= num_samples:
      i = 0
def run_predict_model(model, X_test, batch_size, output_argmax=True):
  '''
  This function runs a prediction on a model that has already been loaded.
  It returns the predicted labels. Meant for multi-class models.
  output_argmax: bool, if True returns the argmaxxed output. def True.
  if output_argmax=False, Y_pred shape is [N_samples,SUBGRID,SUBGRID,SUBGRID,4],
  since it's outputting the Softmax probs the model generates directly
  I: model (keras model), 
  X_test (np.array of shape [N_samples,SUBGRID,SUBGRID,SUBGRID,1]), 
  batch_size (int)
  O: Y_pred (np.array of shape [N_samples,SUBGRID,SUBGRID,SUBGRID,1]) 
  (if output_argmax is True)
  '''
  gen = data_generator(X_test, batch_size)
  N_steps = int(np.ceil(X_test.shape[0] / batch_size))
  Y_pred = []
  for _ in range(N_steps):
    X_batch = next(gen)
    Y_pred.append(model.predict(X_batch, verbose=0))
  Y_pred = np.concatenate(Y_pred, axis=0)
  if output_argmax:
    # if we want the actual predictions [0,1,2,3]
    Y_pred = np.argmax(Y_pred, axis=-1); Y_pred = np.expand_dims(Y_pred, axis=-1)
  return Y_pred
def save_scores_from_model(FILE_DEN, FILE_MSK, FILE_MODEL, FILE_FIG, FILE_PRED, GRID=512, SUBGRID=128, OFF=64, BOXSIZE=205, BOLSHOI_FLAG=False, TRAIN_SCORE=False, COMPILE=False, LATEX=False):
  '''
  Save image of density, mask, and predicted mask. Using save_scores_from_fvol,
  saves F1 scores, confusion matrix to MODEL_NAME_hps.txt and plots confusion matrix.

  FILE_DEN: str density filepath.
  FILE_MSK: str mask filepath.
  FILE_MODEL: str model filepath. should be the same as MODEL_OUT+MODEL_NAME
  FILE_FIG: str where to save figures. default to TNG_multi
  FILE_PRED: str where to save prediction.
  save_4channel: bool before argmax whether or not to save 4-channel prediction
  BOLSHOI_FLAG: bool whether you're working w/ Bolshoi data or not. def TNG mode
  TRAIN_SCORE: bool whether you're scoring on training data or not. def false
  COMPILE: bool whether or not to compile the model. def False to get around
  custom objects error.
  LATEX: bool whether or not to typeset plot labels/titles with LaTeX. added 
  because some envs have thrown errors.
  '''
  MODEL_NAME = FILE_MODEL.split('/')[-1]
  DELTA_NAME = FILE_DEN.split('/')[-1]
  FILE_HPTXT = FILE_MODEL + '_hps.txt'
  BATCH_SIZE = 4 # NOTE can fiddle w this
  ### load model:
  if COMPILE:
    try: 
      model = load_model(FILE_MODEL)
    except IOError:
      print('Model not found. Trying with .keras extension...')
      model = load_model(FILE_MODEL+'.keras')
  else:
    try:
      model = load_model(FILE_MODEL, compile=False)
    except IOError:
      print('Model not found. Trying with .keras extension...')
      model = load_model(FILE_MODEL+'.keras', compile=False)

  X_test = load_dataset(FILE_DEN,SUBGRID,OFF,preproc='mm')
  Y_pred = run_predict_model(model, X_test, BATCH_SIZE)
  Y_pred = assemble_cube2(Y_pred,GRID,SUBGRID,OFF)

  ### write out prediction
  PRED_NAME = MODEL_NAME + '-pred.fvol'
  if BOLSHOI_FLAG == True:
    PRED_NAME = MODEL_NAME + '-pred-bolshoi.fvol'
  volumes.write_fvolume(Y_pred, FILE_PRED+PRED_NAME)
  print(f'Wrote prediction to {FILE_PRED+PRED_NAME}')
  
  # if BOLSHOI, change model name for figure filenames:
  if BOLSHOI_FLAG == True:
    MODEL_NAME = MODEL_NAME + '-bolshoi'

  # if model folder in figs doesnt exist, make it:
  if not os.path.exists(FILE_FIG):
    os.makedirs(FILE_FIG)
    print(f'Created folder {FILE_FIG}')

  ### plot comparison plot of den, mask, pred mask to FILE_FIG:
  den_cmap = 'gray' # default for full DM particle density
  d = volumes.read_fvolume(FILE_DEN); d = d/np.mean(d) + 1e-7 # delta+1
  m = volumes.read_fvolume(FILE_MSK)
  plt.rcParams.update({'font.size': 20})
  fig,ax = plt.subplots(1,3,figsize=(28,12),tight_layout=True)
  i = GRID//3
  if LATEX:
    ax[0].set_title(r'$log(\delta+1)$'+'\n'+f'File: {DELTA_NAME}')
  else:
    ax[0].set_title('Mass Density'+'\n'+f'File: {DELTA_NAME}')
  ax[1].set_title('Predicted Mask')
  ax[2].set_title('True Mask')
  plotter.plot_arr(d,i,ax=ax[0],cmap=den_cmap,logged=True)
  plotter.plot_arr(Y_pred,i,ax=ax[1],segmented_cb=True)
  plotter.plot_arr(m,i,ax=ax[2],segmented_cb=True)
  for axis in ax:
    plotter.set_window(b=0,t=BOXSIZE,Nm=GRID,ax=axis,boxsize=BOXSIZE,Latex=LATEX)
  plt.savefig(FILE_FIG+MODEL_NAME+'-pred-comp.png',facecolor='white',bbox_inches='tight')
  print(f'Saved comparison plot to {FILE_FIG+MODEL_NAME}-pred-comp.png')

  ### plot 3x3 plot of 3 adjacent slices of same comparison^^:
  fig,ax = plt.subplots(3,3,figsize=(28,28),tight_layout=True)
  i = GRID//2
  step = 5
  # i - step slice:
  if LATEX:
    ax[0,0].set_title(r'$log(\delta+1)$'+'\n'+f'File: {DELTA_NAME}')
    ax[1,0].set_title(r'$log(\delta+1)$'+'\n'+f'File: {DELTA_NAME}')
    ax[2,0].set_title(r'$log(\delta+1)$'+'\n'+f'File: {DELTA_NAME}')
  else:
    ax[0,0].set_title('Mass Density'+'\n'+f'File: {DELTA_NAME}')
    ax[1,0].set_title('Mass Density'+'\n'+f'File: {DELTA_NAME}')
    ax[2,0].set_title('Mass Density'+'\n'+f'File: {DELTA_NAME}')

  ax[0,1].set_title(f'Predicted Mask\nSlice {i-step}')
  ax[0,2].set_title(f'True Mask\nSlice {i-step}')
  plotter.plot_arr(d,i-step,ax=ax[0,0],cmap=den_cmap,logged=True)
  plotter.plot_arr(Y_pred,i-step,ax=ax[0,1],segmented_cb=True)
  plotter.plot_arr(m,i-step,ax=ax[0,2],segmented_cb=True)
  # i slice:
  ax[1,1].set_title(f'Predicted Mask\nSlice {i}')
  ax[1,2].set_title(f'True Mask\nSlice {i}')
  plotter.plot_arr(d,i,ax=ax[1,0],cmap=den_cmap,logged=True)
  plotter.plot_arr(Y_pred,i,ax=ax[1,1],segmented_cb=True)
  plotter.plot_arr(m,i,ax=ax[1,2],segmented_cb=True)
  # i + step slice:
  ax[2,1].set_title(f'Predicted Mask\nSlice {i+step}')
  ax[2,2].set_title(f'True Mask\nSlice {i+step}')
  plotter.plot_arr(d,i+step,ax=ax[2,0],cmap=den_cmap,logged=True)
  plotter.plot_arr(Y_pred,i+step,ax=ax[2,1],segmented_cb=True)
  plotter.plot_arr(m,i+step,ax=ax[2,2],segmented_cb=True)
  # fix axis labels to be Mpc/h:
  for axis in ax.flatten():
    plotter.set_window(b=0,t=BOXSIZE,Nm=GRID,ax=axis,boxsize=BOXSIZE,Latex=LATEX)
  plt.savefig(FILE_FIG+MODEL_NAME+'-pred-comp-3x3.png',facecolor='white',bbox_inches='tight')
  print(f'Saved 3x3 comparison plot to {FILE_FIG+MODEL_NAME}-pred-comp-3x3.png')

  # plot vw misclassifications:
  # pick random slice to plot:
  i = np.random.randint(0,GRID)
  fig = plot_vw_misses(m,Y_pred,idx=i,Nm=GRID,boxsize=BOXSIZE)
  fig.savefig(FILE_FIG+MODEL_NAME+f'-pred-comp-vw-miss-slc={i}.png',facecolor='white',bbox_inches='tight')
  # use save_scores_from_fvol to save scores if we want to run the model on its own training data:
  if TRAIN_SCORE == True:
    save_scores_from_fvol(m,Y_pred,FILE_MODEL,FILE_FIG,FILE_DEN,VAL_FLAG=False)

'''
5/1/24: create function to save slices from volumes.
The plots will be identical to those created in save_scores_from_model,
but will not run the prediction in the function
'''
def save_slices_from_fvol(X_test,Y_test,Y_pred,FILE_MODEL,FILE_FIG,lamb,BOXSIZE=205,GRID=512,SUBGRID=128,OFF=64,BOLSHOI_FLAG=False,LATEX=False):
  '''
  X_test: np.ndarray of shape (N_samples,SUBGRID,SUBGRID,SUBGRID,1). Should be in [0,1] range
  Y_test: np.ndarray of shape (N_samples,SUBGRID,SUBGRID,SUBGRID,1). Should be int labels [0,1,2,3]
  Y_pred: np.ndarray of shape (N_samples,SUBGRID,SUBGRID,SUBGRID,1). Should be int labels [0,1,2,3]
  (this means that if you run nets.run_predict_model(output_argmax=False), you need to one-hot encode)
  FILE_MODEL: str model filepath. should be the same as MODEL_OUT+MODEL_NAME
  FILE_FIG: str where to save figures. figs will be saved to FILE_FIG + MODEL_NAME/
  GRID: int, size of whole cube on a side. def 512
  SUBGRID: int, size of subcubes on a side. def 128
  OFF: int. size of overlap for each subcube. def 64.
  if running test case of GRID=128, SUBGRID=32, OFF=16
  BOLSHOI_FLAG: bool, whether volume is from Bolshoi or not. added to keep track of figure filenames
  when running a model trained on TNG on Bolshoi data for validation.
  LATEX: bool, whether or not to typeset axes labels and titles using LaTeX.

  NOTE that this will not work for validation data since it has been shuffled and is not
  in the right shape for assemble_cube2. This function is mostly meant for 45 deg rotated
  plots for validation.
  '''
  MODEL_NAME = FILE_MODEL.split('/')[-1]
  d = assemble_cube2(X_test,GRID,SUBGRID,OFF) # density field
  d += 1e-7 # suppress div by zero warnings
  m = assemble_cube2(Y_test,GRID,SUBGRID,OFF) # mask field
  p = assemble_cube2(Y_pred,GRID,SUBGRID,OFF) # predicted mask field
  # if BOLSHOI, change model name for figure filenames:
  if BOLSHOI_FLAG == True:
    MODEL_NAME = MODEL_NAME + '-bolshoi'
  # if model folder in figs doesnt exist, make it:
  if not os.path.exists(FILE_FIG):
    os.makedirs(FILE_FIG)
    print(f'Created folder {FILE_FIG}')
  # PLOTTING:
  den_cmap = 'gray' # default for full DM particle density
  plt.rcParams.update({'font.size': 20})
  fig,ax = plt.subplots(1,3,figsize=(28,12),tight_layout=True)
  i = GRID//3
  if LATEX:
    ax[0].set_title(r'$log(\delta+1)$'+'\n'+fr'$\lambda={lamb}$')
  else:
    ax[0].set_title('Mass Density'+'\n'+f'L={lamb}')
  ax[1].set_title('Predicted Mask')
  ax[2].set_title('True Mask')
  plotter.plot_arr(d,i,ax=ax[0],cmap=den_cmap,logged=True)
  plotter.plot_arr(p,i,ax=ax[1],segmented_cb=True)
  plotter.plot_arr(m,i,ax=ax[2],segmented_cb=True)
  for axis in ax:
    plotter.set_window(b=0,t=BOXSIZE,Nm=GRID,ax=axis,boxsize=BOXSIZE,Latex=LATEX)
  plt.savefig(FILE_FIG+MODEL_NAME+'-pred-comp.png',facecolor='white',bbox_inches='tight')
  print(f'Saved comparison plot to {FILE_FIG+MODEL_NAME}-pred-comp.png')

  ### plot 3x3 plot of 3 adjacent slices of same comparison^^:
  fig,ax = plt.subplots(3,3,figsize=(28,28),tight_layout=True)
  i = GRID//2
  step = 10
  if LATEX:
    ax[0,0].set_title(r'$log(\delta+1)$'+'\n'+fr'$\lambda={lamb}$')
    ax[1,0].set_title(r'$log(\delta+1)$'+'\n'+fr'$\lambda={lamb}$')
    ax[2,0].set_title(r'$log(\delta+1)$'+'\n'+fr'$\lambda={lamb}$')
  else:
    ax[0,0].set_title('Mass Density'+'\n'+f'L={lamb}')
    ax[1,0].set_title('Mass Density'+'\n'+f'L={lamb}')
    ax[2,0].set_title('Mass Density'+'\n'+f'L={lamb}')
  # i - step slice:
  ax[0,1].set_title(f'Predicted Mask\nSlice {i-step}')
  ax[0,2].set_title(f'True Mask\nSlice {i-step}')
  plotter.plot_arr(d,i-step,ax=ax[0,0],cmap=den_cmap,logged=True)
  plotter.plot_arr(p,i-step,ax=ax[0,1],segmented_cb=True)
  plotter.plot_arr(m,i-step,ax=ax[0,2],segmented_cb=True)
  # i slice:
  ax[1,1].set_title(f'Predicted Mask\nSlice {i}')
  ax[1,2].set_title(f'True Mask\nSlice {i}')
  plotter.plot_arr(d,i,ax=ax[1,0],cmap=den_cmap,logged=True)
  plotter.plot_arr(p,i,ax=ax[1,1],segmented_cb=True)
  plotter.plot_arr(m,i,ax=ax[1,2],segmented_cb=True)
  # i + step slice:
  ax[2,1].set_title(f'Predicted Mask\nSlice {i+step}')
  ax[2,2].set_title(f'True Mask\nSlice {i+step}')
  plotter.plot_arr(d,i+step,ax=ax[2,0],cmap=den_cmap,logged=True)
  plotter.plot_arr(p,i+step,ax=ax[2,1],segmented_cb=True)
  plotter.plot_arr(m,i+step,ax=ax[2,2],segmented_cb=True)
  # fix axis labels to be Mpc/h:
  for axis in ax.flatten():
    plotter.set_window(b=0,t=BOXSIZE,Nm=GRID,ax=axis,boxsize=BOXSIZE,Latex=LATEX)
  plt.savefig(FILE_FIG+MODEL_NAME+'-pred-comp-3x3.png',facecolor='white',bbox_inches='tight')
  print(f'Saved 3x3 comparison plot to {FILE_FIG+MODEL_NAME}-pred-comp-3x3.png')
# adding function that creates a MODEL_NAME from some parameters:
def create_model_name(SIM, DEPTH, FILTERS, GRID, LAMBDA_TH, SIGMA, base_L, UNIFORM_FLAG, BN_FLAG, DROP, LOSS, TL_TYPE=None, tran_L=None, suffix=None):
  '''
  Create a model name from the parameters of the model:
  SIM: str, simulation model was trained on
  DEPTH: int, depth of U-net
  FILTERS: int, number of filters in first layer
  GRID: int, size of full cube on a side
  LAMBDA_TH: float, threshold value for eigenvalue definition. normally 0.65
  SIGMA: float, sigma value for Gaussian smoothing. normally 2.4 for TNG and 0.916 for Bolshoi
  base_L: float, interparticle spacing model was trained on originally
  UNIFORM_FLAG: bool, whether model uses uniform random sampling. def False
  BN_FLAG: bool, whether model uses batch normalization. def False
  DROP: float, dropout rate. 0.0 means no dropout.
  LOSS: str, loss function model was trained with
  TL_TYPE: str, type of transfer learning model was trained with, if any. def None,
  possible values: 'ENC', 'LL', 'ENC_EO'
  tran_L: float, interparticle spacing model was transfer learned to, if it was transfer learned, def None
  suffix: str, any additional suffix to add to model name. def None
  '''
  if SIM == 'BOL':
    SIM = 'Bolshoi'
  # root name:
  mn = f'{SIM}_D{DEPTH}-F{FILTERS}-Nm{GRID}-th{LAMBDA_TH}-sig{SIGMA}-base_L{base_L}'
  # add flags:
  if UNIFORM_FLAG:
    mn += '_uniform'
  if BN_FLAG:
    mn += '_BN'
  if DROP != 0.0:
    mn += f'_DROP{DROP}'
  if LOSS == 'SCCE':
    mn += '_SCCE'
  if LOSS == 'FOCAL_CCE':
    mn += '_FOCAL'
  if LOSS == 'DISCCE':
    mn += '_DISCCE'
  if LOSS == 'CCE':
    pass
  if suffix is not None:
    mn += f'_{suffix}'
  if TL_TYPE is not None:
    print('Model was transfer learned. Adjusting name...')
    mn += f'_TL{TL_TYPE}-tran_L{tran_L}'
  return mn
# adding function to spit out all possible parameters from a model name
def parse_model_name(MODEL_NAME):
  '''
  Input: MODEL_NAME: str, name of model. can be base model or transfer
  learned.
  Output: dictionary of model parameters.
  SIM: str, simulation model was trained on
  DEPTH: int, depth of model
  FILTERS: int, number of filters in first layer
  GRID: int, size of full cube
  Lambda_th: float, threshold value for eigenvalue definition
  sig: float, sigma value for Gaussian smoothing
  BN: bool, whether model uses batch normalization
  DROP: float, dropout rate. 0.0 means no dropout.
  UNIFORM_FLAG: bool, whether model uses uniform random sampling. def False
  LOSS: str, loss function model was trained with
  TL_TYPE: str, type of transfer learning model was trained with (if any)
  base_L: int, interparticle spacing model was trained on originally
  tran_L: int, interparticle spacing model was transfer learned to
  example base MODEL_NAME:
  TNG_D4-F16-Nm256-th0.65-sig1.2-base_L3_SCCE
  example TLed MODEL_NAME:
  TNG_D3-F32-Nm512-th0.65-sig2.4-base_L0.33-tran_L3_SCCE
  '''
  SIM = MODEL_NAME.split('_')[0]
  DEPTH = int(MODEL_NAME.split('_')[1][1])
  FILTERS = int(MODEL_NAME.split('-F')[1].split('-')[0])
  GRID = int(MODEL_NAME.split('Nm')[1].split('-')[0])
  LAMBDA_TH = float(MODEL_NAME.split('-th')[1].split('-')[0])
  SIGMA = float(MODEL_NAME.split('-sig')[1].split('-')[0])
  base_L = float(MODEL_NAME.split('base_L')[1][0])
  if base_L == 0.0 and SIM == 'TNG':
    base_L = 0.33
  if base_L == 0.0 and SIM == 'Bolshoi':
    base_L = 0.122
  BN_FLAG = 'BN' in MODEL_NAME
  UNIFORM_FLAG = 'uniform' in MODEL_NAME
  if 'DROP' in MODEL_NAME:
    DROP = float(MODEL_NAME.split('DROP')[1].split('_')[0])
  else:
    DROP = 0.0
  if 'TL' in MODEL_NAME:
    TL_TYPE = MODEL_NAME.split('TL')[1].split('_')[0]
    tran_L = MODEL_NAME.split('tran_L')[1]
  else:
    TL_TYPE = None
    tran_L = None
  if 'SCCE' in MODEL_NAME:
    LOSS = 'SCCE'
  elif 'FOCAL' in MODEL_NAME:
    LOSS = 'FOCAL_CCE'
  else:
    LOSS = 'CCE'
  if 'DISCCE' in MODEL_NAME:
    LOSS = 'DISCCE'
  # return dictionary of parameters:
  return {
    'SIM': SIM,
    'DEPTH': DEPTH,
    'FILTERS': FILTERS,
    'GRID': GRID,
    'LAMBDA_TH': LAMBDA_TH,
    'SIGMA': SIGMA,
    'BN': BN_FLAG,
    'DROP': DROP,
    'UNIFORM_FLAG': UNIFORM_FLAG,
    'LOSS': LOSS,
    'TL_TYPE': TL_TYPE,
    'base_L': base_L,
    'tran_L': tran_L
  }
# adding fxn that gets layer index from its name, for freezing purposes:
def get_layer_index(model, layer_name):
  '''
  model: keras model
  layer_name: str, name of layer to get index of
  '''
  for i, layer in enumerate(model.layers):
    if layer.name == layer_name:
      return i
  return None

# adding a function that already existed i think?
def load_dict_from_text(file_path,string_break='total_params'):
  '''
  Load a dictionary from a text file. 
  file_path: str, path to text file
  string_break: str, string to break at. def 'total_params'
  '''
  dict_out = dict()
  with open(file_path, 'r') as f:
    for line in f:
      print(line)
      if line.split(':')[0] == string_break:
        break
      else:
        dict_out[line.split(':')[0]] = line.split(':')[1].strip()
  return dict_out

# adding generator functions for tf.data.Dataset
def data_gen_mmap(FILE_X,FILE_Y):
  '''
  Generator function to feed data in batches to the model.
  Helps with OOM errors when training on a large volume.
  Assumes that FILE_X_TRAIN and FILE_Y_TRAIN are .npy files.
  FILE_X: str, filepath to X_(train/test) data
  FILE_Y: str, filepath to Y_(train/test) data
  '''
  X = np.load(FILE_X,mmap_mode='r')
  Y = np.load(FILE_Y,mmap_mode='r')
  for features, labels in zip(X,Y):
    yield (features, labels)

# creating better generator function for tf.data.Dataset
def data_gen_mmap_batch(FILE_X,FILE_Y,batch_size):
  '''
  NOTE THIS DOESN'T WORK: adds another None dim???
  Generator function to feed data in batches to the model.
  Helps with OOM errors when training on a large volume.
  Assumes that FILE_X_TRAIN and FILE_Y_TRAIN are .npy files.
  FILE_X: str, filepath to X_(train/test) data
  FILE_Y: str, filepath to Y_(train/test) data
  '''
  X = np.memmap(FILE_X, dtype='float32', mode='r')
  Y = np.memmap(FILE_Y, dtype='int8', mode='r')
  num_samples = X.shape[0]
  for start_idx in range(0,num_samples,batch_size):
    end_idx = min(start_idx+batch_size,num_samples)
    yield (X[start_idx:end_idx], Y[start_idx:end_idx])