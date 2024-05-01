#!/usr/bin/env python3
'''
3/17/24: Making an updated version of the nets.py file.
Importing less random stuff and making it more lightweight and readable.

This is meant to be used for multi-class classification.
The old nets.py is fine for binary.
'''
import gc
import os
import sys
import csv
sys.path.append('/ifs/groups/vogeleyGrp/nets/')
import volumes
import plotter
from collections import Counter
import matplotlib.pyplot as plt
class_labels = ['Void','Wall','Filament','Halo']
import numpy as np
from scipy import ndimage as ndi
from keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.utils.layer_utils import count_params # doesnt work?
from tensorflow.python.keras.utils import layer_utils
from keras.utils import to_categorical
from keras.layers import Input, Conv3D, MaxPooling3D, Conv3DTranspose, UpSampling3D, Concatenate, BatchNormalization, Activation, Dropout
from keras import backend as K
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger

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
  vals,cnts = np.unique(msk,return_counts=True)
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
  X_all = X_all.astype('float16')
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
  print(f'Reading mask: {FILE_MASK}...')
  msk = volumes.read_fvolume(FILE_MASK)
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
  return X_all_overlap, Y_all_overlap

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
    X_all = X_all.astype('int8')
  else:
    X_all = X_all.astype('float16')
  gc.collect()
  return X_all
#---------------------------------------------------------
# Focal loss function
#---------------------------------------------------------
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

  def categorical_focal_loss_fixed(y_true, y_pred):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred: A tensor resulting from a softmax
    :return: Output tensor.
    """

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
# U-Net creator
#---------------------------------------------------------
def conv_block(input_tensor, filters, activation='relu', batch_normalization=True, dropout_rate=None):
  x = Conv3D(filters, kernel_size=(3, 3, 3), padding='same')(input_tensor)
  if batch_normalization:
      x = BatchNormalization()(x)
  x = Activation(activation)(x)
  if dropout_rate:
      x = Dropout(dropout_rate)(x)
  x = Conv3D(filters, kernel_size=(3, 3, 3), padding='same')(x)
  if batch_normalization:
      x = BatchNormalization()(x)
  x = Activation(activation)(x)
  return x
#---------------------------------------------------------
def unet_3d(input_shape, num_classes, initial_filters=32, depth=4, activation='relu', last_activation='softmax', batch_normalization=False, dropout_rate=None, model_name='3D_U_Net', report_params=False):
  inputs = Input(input_shape)
  
  # Encoder path
  encoder_outputs = []
  x = inputs
  for d in range(depth):
    filters = initial_filters * (2 ** d)
    x = conv_block(x, filters, activation, batch_normalization, dropout_rate)
    encoder_outputs.append(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
  
  # Bottom
  x = conv_block(x, initial_filters * (2 ** depth), activation, batch_normalization, dropout_rate)
  
  # Decoder path
  for d in reversed(range(depth)):
    filters = initial_filters * (2 ** d)
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Concatenate(axis=-1)([x, encoder_outputs[d]])
    x = conv_block(x, filters, activation, batch_normalization, dropout_rate)
  
  # Output
  outputs = Conv3D(num_classes, kernel_size=(1, 1, 1))(x)
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
  '''
  #FILE_HPTXT = FILE_MODEL + '_hps.txt'
  MODEL_NAME = FILE_MODEL.split('/')[-1]
  # calculate F1 scores:
  ps, rs, f1s, _ = precision_recall_fscore_support(y_true.ravel(), y_pred.ravel(), average=None,zero_division=0.0)
  micro_f1 = f1_score(y_true.ravel(), y_pred.ravel(), average='micro')
  macro_f1 = f1_score(y_true.ravel(), y_pred.ravel(), average='macro')
  weight_f1 = f1_score(y_true.ravel(), y_pred.ravel(), average='weighted')
  # NOTE WRITING SCORES TO HYPERPARAMETER TXT FILES IS DEPRECATED!
  #with open(FILE_HPTXT, 'a') as f:
  #  for i in range(len(f1s)):
  #    f.write(f'Class {class_labels[i]} F1: {f1s[i]} \n')
  #  f.write(f'\nAverage F1: {np.mean(f1s)} \n')
  # add to score_dict:
  score_dict['micro_f1'] = micro_f1
  score_dict['macro_f1'] = macro_f1
  score_dict['weighted_f1'] = weight_f1
  for i in range(len(f1s)):
    score_dict[f'class_{class_labels[i]}_f1'] = f1s[i]
    score_dict[f'class_{class_labels[i]}_precision'] = ps[i]
    score_dict[f'class_{class_labels[i]}_recall'] = rs[i]
    print(f'Class {class_labels[i]} F1: {f1s[i]}')
    print(f'Class {class_labels[i]} precision: {ps[i]}')
    print(f'Class {class_labels[i]} recall: {rs[i]}')
  print(f'Micro F1: {micro_f1} \nMacro F1: {macro_f1} \nWeighted F1: {weight_f1}')

  
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
     
def ROC_curves(y_true, y_pred, FILE_MODEL, FILE_FIG, score_dict, micro=True, macro=True):
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
  N_classes = len(class_labels)
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

# 7/19/23: updated this to use helper functions instead.
def save_scores_from_fvol(y_true, y_pred, FILE_MODEL, FILE_FIG, score_dict, VAL_FLAG=True, downsample=10):
  '''
  Save F1 scores, confusion matrix, population histograms, ROC curves,
  precision-recall curves. THIS FXN DOES NOT PREDICT!
  Inputs:
  y_true: true labels 
  (shape=[Nm,Nm,Nm], or [N_samples,SUBGRID,SUBGRID,SUBGRID,1])
  y_pred: predicted class PROBABILITIES NOTE: used to be actual preds
  (shape=[Nm,Nm,Nm,4], or [N_samples,SUBGRID,SUBGRID,SUBGRID,4])
  FILE_MODEL: str model filepath. should be the same as MODEL_OUT+MODEL_NAME
  FILE_FIG: str filepath to save figures
  score_dict: dict to save scores to
  VAL_FLAG: bool, True if scores are based on val data, False if not. def True
  downsample: int, skip parameter to downsample for ROC, PR curves. def 10.
  e.g. if downsample = 100, every 100th voxel is considered
  '''
  if y_pred.shape[-1] != 4:
    print('y_pred must be a 4 channel array of class probabilities. save_scores_from_fvol may not work as intended')
  # get in shape for ROC, PR curves:
  y_true_binarized = to_categorical(y_true,num_classes=4,dtype='int8')
  y_true_binarized = y_true_binarized.reshape(-1,4)
  y_pred_reshaped = y_pred.reshape(-1,4)
  # downsample by some skip parameter, e.g. 10:
  y_true_binarized = y_true_binarized[::downsample]
  y_pred_reshaped =   y_pred_reshaped[::downsample]
  ### REQUIRES DIRECT SOFTMAX OUTPUT PROBABILITIES ###
  ROC_curves(y_true_binarized, y_pred_reshaped, FILE_MODEL, FILE_FIG, score_dict)
  PR_curves( y_true_binarized, y_pred_reshaped, FILE_MODEL, FILE_FIG, score_dict)
  # get in shape for F1s, Confusion matrix:
  y_pred = np.argmax(y_pred, axis=-1); y_pred = np.expand_dims(y_pred, axis=-1)
  # now y_true and y_pred both have shape [N_samples,SUBGRID,SUBGRID,SUBGRID,1]
  ### REQUIRES ARG-MAXXED CLASS PREDICTIONS ###
  F1s(y_true, y_pred, FILE_MODEL, score_dict)
  CMatrix(y_true, y_pred, FILE_MODEL, FILE_FIG)
  #population_hists(y_true, y_pred, FILE_MODEL, FILE_FIG, FILE_DEN) # broken rn w/ test data
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
def save_scores_to_csv(score_dict, file_path):
  '''
  score_dict: dict of scores. 
  filepath: str of where to save scores.csv

  each dict of scores will be APPENDED to the csv, not overwriting. 
  '''
  file_exists = os.path.isfile(file_path)
  with open(file_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=score_dict.keys())
    if not file_exists:
      writer.writeheader()
    writer.writerow(score_dict)
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
  BATCH_SIZE = 8 # NOTE can fiddle w this
  ### load model:
  if COMPILE:
    model = load_model(FILE_MODEL)
  else:
    model = load_model(FILE_MODEL, compile=False)

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
  d = volumes.read_fvolume(FILE_DEN); d = d/np.mean(d) # delta+1
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

  # use save_scores_from_fvol to save scores if we want to run the model on its own training data:
  if TRAIN_SCORE == True:
    save_scores_from_fvol(m,Y_pred,FILE_MODEL,FILE_FIG,FILE_DEN,VAL_FLAG=False)
