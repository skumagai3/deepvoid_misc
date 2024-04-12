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
sys.path.append('/ifs/groups/vogeleyGrp/nets/')
import volumes
import plotter
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
from keras.losses import CategoricalCrossentropy
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

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
def load_dataset_all(FILE_DEN, FILE_MASK, SUBGRID, preproc, classification=True, sigma=None, binary_mask=False):
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
    summary(den); summary(msk)
  if preproc == 'std':
    den = standardize(den)
    #msk = standardize(msk)
    print('Ran preprocessing by dividing density/mask by std dev and subtracting by the mean! ')
    print('\nNew summary statistics: ')
    summary(den); summary(msk)
  # Make wall mask
  #msk = np.zeros(den_shp,dtype=np.uint8)
  n_bins = den_shp[0] // SUBGRID

  cont = 0 
  X_all = np.zeros(shape=((n_bins**3)*4, SUBGRID,SUBGRID,SUBGRID,1))
  if classification == False:
    Y_all = np.ndarray(((n_bins**3)*4, SUBGRID,SUBGRID,SUBGRID,1),dtype=np.float32)
  else:
    Y_all = np.ndarray(((n_bins**3)*4, SUBGRID,SUBGRID,SUBGRID,1),dtype=np.int32)

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
  return X_all, Y_all
#---------------------------------------------------------
# For loading testing/validation data for prediction
#---------------------------------------------------------
def load_dataset(file_in, SUBGRID, OFF, preproc='mm',sigma=None):
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
  nbins = (den.shape[0] // SUBGRID) + 1 + 1 + 1
  if den.shape[0] == 640:
    nbins += 1
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
      
  X_all = X_all.astype('float16')
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
  '''
  def __init__(self,val_data,N_epochs,avg='weighted',beta=1.0):
    super().__init__()
    self.validation_data = val_data
    self.N_epochs = N_epochs
    self.avg = avg
    self.beta = beta
  def on_epoch_end(self,epoch,logs={}):
    if epoch % self.N_epochs == 0:
      X_test = self.validation_data[0]; Y_test = self.validation_data[1]
      Y_pred = self.model.predict(X_test,verbose=0)
      _val_loss, _val_acc = self.model.evaluate(X_test,Y_test,verbose=0)
      _val_ROC_AUC = roc_auc_score(Y_test.reshape(-1,4),Y_pred.reshape(-1,4),average=self.avg,multi_class='ovr')
      Y_test = np.argmax(Y_test,axis=-1); Y_test = np.expand_dims(Y_test,axis=-1)
      Y_pred = np.argmax(Y_pred,axis=-1); Y_pred = np.expand_dims(Y_pred,axis=-1)
      Y_test = Y_test.flatten(); Y_pred = Y_pred.flatten()
      _val_balanced_acc = balanced_accuracy_score(Y_test,Y_pred)
      _val_precision, _val_recall, _val_f1, _val_support = precision_recall_fscore_support(Y_test,Y_pred,beta=self.beta,average=self.avg,zero_division=0.0)
      _val_matt_corrcoef = matthews_corrcoef(Y_test,Y_pred)
      _val_void_precision, _val_void_recall, _val_void_f1, _val_void_support = precision_recall_fscore_support(Y_test,Y_pred,beta=self.beta,average=self.avg,zero_division=0.0)

      logs['val_loss'] = _val_loss
      logs['val_acc'] = _val_acc
      logs['val_balanced_acc'] = _val_balanced_acc
      logs['val_f1'] = _val_f1
      logs['val_recall'] = _val_recall
      logs['val_precision'] = _val_precision
      logs['val_ROC_AUC'] = _val_ROC_AUC
      logs['val_matt_corrcoef'] = _val_matt_corrcoef
      logs['val_void_f1'] = _val_void_f1
      logs['val_void_recall'] = _val_void_recall
      logs['val_void_precision'] = _val_void_precision
      gc.collect()

      #print(f' - Balanced Acc: {_val_balanced_acc:.4f} - F1: {_val_f1:.4f} - Precision: {_val_precision:.4f} - Recall: {_val_recall:.4f} - ROC AUC: {_val_ROC_AUC:.4f} \nMatt Corr Coef: {_val_matt_corrcoef:.4f} - Void F1: {_val_void_f1:.4f} - Void Recall: {_val_void_recall:.4f} - Void Precision: {_val_void_precision:.4f}')
      return
#---------------------------------------------------------
# Scoring functions for multi-class classification
#---------------------------------------------------------

def F1s(y_true, y_pred, FILE_MODEL):
  '''
  helper fxn for save_scores_from_fvol to calculate F1 scores
  and write to hyperparameters txt file.
  '''
  FILE_HPTXT = FILE_MODEL + '_hps.txt'
  MODEL_NAME = FILE_MODEL.split('/')[-1]
  # calculate F1 scores:
  f1s = f1_score(y_true.flatten(), y_pred.flatten(), average=None)
  with open(FILE_HPTXT, 'a') as f:
    for i in range(len(f1s)):
      f.write(f'Class {class_labels[i]} F1: {f1s[i]} \n')
    f.write(f'\nAverage F1: {np.mean(f1s)} \n')

def CMatrix(y_true, y_pred, FILE_MODEL, FILE_FIG):
  '''
  helper fxn for save_scores_from_fvol to plot confusion matrix
  and save a text version in hyperparameters txt file.
  '''
  FILE_HPTXT = FILE_MODEL + '_hps.txt'
  MODEL_NAME = FILE_MODEL.split('/')[-1]
  # compute confusion matrix:
  plt.rcParams.update({'font.size': 14})
  cm = confusion_matrix(y_true.flatten(), y_pred.flatten(),
                        labels=[0,1,2,3],normalize='true')
  class_report = classification_report(y_true.flatten(), y_pred.flatten(),labels=class_labels,output_dict=True)
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
     
def ROC_curves(y_true, y_pred, FILE_MODEL, FILE_FIG):
  '''
  Helper function for save_scores_from_fvol to plot ROC curves.
  '''
  FILE_HPTXT = FILE_MODEL + '_hps.txt'
  MODEL_NAME = FILE_MODEL.split('/')[-1]
  # plot ROC curves:
  plt.rcParams.update({'font.size': 16})
  fig, ax = plt.subplots(1,1,figsize=(8,8))
  ax.axis('square')
  ax.set_title('Class OvR ROC Curves')
  ax.plot([0,1],[0,1],linestyle='--',lw=2,color='r',label='Chance level')
  aucs = []
  for i in range(len(class_labels)):
    binary_m = (y_true == i).flatten(); binary_p = (y_pred == i).flatten()
    fpr, tpr, threshs = roc_curve(binary_m, binary_p)
    roc_auc = auc(fpr, tpr); aucs.append(roc_auc)
    display = RocCurveDisplay(fpr=fpr,tpr=tpr,
                              roc_auc=roc_auc,
                              estimator_name=class_labels[i]+' OvR')
    display.plot(ax=ax)
  ax.legend(loc='best',prop={'size': 11})
  ax.set_xlim(-0.05,1.05); ax.set_ylim(-0.05,1.05)
  plt.savefig(FILE_FIG+MODEL_NAME+'_ROC_OvR.png',facecolor='white',bbox_inches='tight')
  print(f'Saved ROC OvR curves for each class to '+FILE_FIG+MODEL_NAME+'_ROC_OvR.png')
  with open(FILE_HPTXT, 'a') as f:
    f.write('\n'+f'Average AUC: {np.mean(aucs)}\n')
    for i in range(len(aucs)):
      f.write('\n'+f'Class {class_labels[i]} AUC: {aucs[i]}\n')

def PR_curves(y_true, y_pred, FILE_MODEL, FILE_FIG):
  '''
  helper function for save_scores_from_fvol to plot Prec-Recall curves.
  '''
  FILE_HPTXT = FILE_MODEL + '_hps.txt'
  MODEL_NAME = FILE_MODEL.split('/')[-1]
  # calc prec, recall, avg prec:
  precision = {}; recall = {}; avg_prec = {}
  for i in range(len(class_labels)):
    precision[i], recall[i], _ = precision_recall_curve((y_true==i).flatten(), 
                                                         (y_pred==i).flatten())
    avg_prec[i] = average_precision_score((y_true==i).flatten(), (y_pred==i).flatten())
  # plot PR curves:
  plt.rcParams.update({'font.size': 16})
  fig, ax = plt.subplots(1,1,figsize=(8,8))
  f_scores = np.linspace(0.3, 0.9, num=4)
  lines, labels = [], []
  for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    (l,) = ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    ax.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02),
                fontsize=10)
  for i in range(len(class_labels)):
    display = PrecisionRecallDisplay(precision=precision[i], recall=recall[i],
                                     average_precision=avg_prec[i])
    display.plot(ax=ax,name=class_labels[i]+' PR')
  handles, labels = ax.get_legend_handles_labels()
  handles.extend([l])
  labels.extend(["iso-F1 curves"])
  ax.set_xlim([0.0,1.0]); ax.set_ylim([0.0,1.05])
  ax.legend(handles=handles, labels=labels, loc='best',prop={'size': 10})
  ax.set_title('Multi-Label Precision-Recall Curves')
  plt.savefig(FILE_FIG+MODEL_NAME+'_PR.png',facecolor='white',bbox_inches='tight')
  print('Saved precision-recall curves for each class at: '+FILE_FIG+MODEL_NAME+'_PR.png')
# 7/19/23: updated this to use helper functions instead.
def save_scores_from_fvol(y_true, y_pred, FILE_MODEL, FILE_FIG, FILE_DEN):
  '''
  Save F1 scores, confusion matrix, population histograms, ROC curves,
  precision-recall curves. THIS FXN DOES NOT PREDICT!
  Inputs:
  y_true: true labels (shape=[Nm,Nm,Nm])
  y_pred: predicted class labels (shape=[Nm,Nm,Nm])
  FILE_MODEL: str model filepath. should be the same as MODEL_OUT+MODEL_NAME
  '''
  F1s(y_true, y_pred, FILE_MODEL)
  CMatrix(y_true, y_pred, FILE_MODEL, FILE_FIG)
  #population_hists(y_true, y_pred, FILE_MODEL, FILE_FIG, FILE_DEN) # broken rn w/ test data
  ROC_curves(y_true, y_pred, FILE_MODEL, FILE_FIG)
  PR_curves(y_true, y_pred, FILE_MODEL, FILE_FIG)
  print('Saved metrics.')
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
def run_predict_model(model, X_test, batch_size):
  '''
  This function runs a prediction on a model that has already been loaded.
  It returns the predicted labels. Meant for multi-class models.
  I: model (keras model), 
  X_test (np.array of shape [N_samples,SUBGRID,SUBGRID,SUBGRID,1]), 
  batch_size (int)
  O: Y_pred (np.array of shape [N_samples,SUBGRID,SUBGRID,SUBGRID,1])
  '''
  gen = data_generator(X_test, batch_size)
  N_steps = int(np.ceil(X_test.shape[0] / batch_size))
  Y_pred = []
  for _ in range(N_steps):
    X_batch = next(gen)
    Y_pred.append(model.predict(X_batch, verbose=0))
  Y_pred = np.concatenate(Y_pred, axis=0)
  Y_pred = np.argmax(Y_pred, axis=-1); Y_pred = np.expand_dims(Y_pred, axis=-1)
  return Y_pred
def save_scores_from_model(FILE_DEN, FILE_MSK, FILE_MODEL, FILE_FIG, GRID=512, BOXSIZE=205, BOLSHOI_FLAG=False, TRAIN_SCORE=False, COMPILE=False):
  '''
  Save image of density, mask, and predicted mask. Using save_scores_from_fvol,
  saves F1 scores, confusion matrix to MODEL_NAME_hps.txt and plots confusion matrix.

  FILE_DEN: str density filepath.
  FILE_MSK: str mask filepath.
  FILE_MODEL: str model filepath. should be the same as MODEL_OUT+MODEL_NAME
  FILE_FIG: str where to save figures. default to TNG_multi
  save_4channel: bool before argmax whether or not to save 4-channel prediction
  BOLSHOI_FLAG: bool whether you're working w/ Bolshoi data or not. def TNG mode
  TRAIN_SCORE: bool whether you're scoring on training data or not. def false
  COMPILE: bool whether or not to compile the model. def False to get around
  custom objects error.
  '''
  SUBGRID = 128; OFF = 64
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
  Y_pred, Y_test = run_predict_model(model, X_test, Y_test, BATCH_SIZE)
  Y_pred = assemble_cube2(Y_pred,GRID,SUBGRID,OFF)

  ### write out prediction
  PRED_NAME = MODEL_NAME + '-pred.fvol'
  if BOLSHOI_FLAG == True:
    PRED_NAME = MODEL_NAME + '-pred-bolshoi.fvol'
  volumes.write_fvolume(Y_pred, '/ifs/groups/vogeleyGrp/nets/preds/'+PRED_NAME)
  print(f'Wrote prediction to /ifs/groups/vogeleyGrp/nets/preds/{PRED_NAME}')
  
  # if BOLSHOI, change model name for figure filenames:
  if BOLSHOI_FLAG == True:
    MODEL_NAME = MODEL_NAME + '-bolshoi'

  # if model folder in figs doesnt exist, make it:
  if not os.path.exists(FILE_FIG):
    os.makedirs(FILE_FIG)
    print(f'Created folder {FILE_FIG}')

  ### plot comparison plot of den, mask, pred mask to FILE_FIG:
  #FILE_MSK = '/ifs/groups/vogeleyGrp/data/TNG/alonso_mask_th=0.65_sig=2.3.fvol'
  den_cmap = 'gray' # default for full DM particle density
  if FILE_DEN != '/ifs/groups/vogeleyGrp/data/TNG/DM_DEN_snap99_Nm=512.fvol':
    den_cmap = 'gray_r'
  d = volumes.read_fvolume(FILE_DEN); d = d/np.mean(d) # delta+1
  m = volumes.read_fvolume(FILE_MSK)
  plt.rcParams.update({'font.size': 20})
  fig,ax = plt.subplots(1,3,figsize=(28,12),tight_layout=True)
  i = 300
  ax[0].set_title(r'$log(\delta+1)$'+'\n'+f'File: {DELTA_NAME}')
  ax[1].set_title('Predicted Mask')
  ax[2].set_title('True Mask')
  plotter.plot_arr(d,i,ax=ax[0],cmap=den_cmap,logged=True)
  plotter.plot_arr(Y_pred,i,ax=ax[1],cmap='viridis',cb=False)
  plotter.plot_arr(m,i,ax=ax[2],cmap='viridis',cb=False)
  for axis in ax:
    plotter.set_window(0,BOXSIZE,GRID,axis,BOXSIZE)
  plt.savefig(FILE_FIG+MODEL_NAME+'-pred-comp.png',facecolor='white',bbox_inches='tight')
  print(f'Saved comparison plot to {FILE_FIG+MODEL_NAME}-pred-comp.png')

  ### plot 3x3 plot of 3 adjacent slices of same comparison^^:
  fig,ax = plt.subplots(3,3,figsize=(28,28),tight_layout=True)
  i = 200
  step = 5
  # i - step slice:
  ax[0,0].set_title(r'$log(\delta+1)$'+'\n'+f'File: {DELTA_NAME}')
  ax[0,1].set_title(f'Predicted Mask\nSlice {i-step}')
  ax[0,2].set_title(f'True Mask\nSlice {i-step}')
  plotter.plot_arr(d,i-step,ax=ax[0,0],cmap=den_cmap,logged=True)
  plotter.plot_arr(Y_pred,i-step,ax=ax[0,1],cmap='viridis',cb=False)
  plotter.plot_arr(m,i-step,ax=ax[0,2],cmap='viridis',cb=False)
  # i slice:
  ax[1,0].set_title(r'$log(\delta+1)$'+'\n'+f'File: {DELTA_NAME}')
  ax[1,1].set_title(f'Predicted Mask\nSlice {i}')
  ax[1,2].set_title(f'True Mask\nSlice {i}')
  plotter.plot_arr(d,i,ax=ax[1,0],cmap=den_cmap,logged=True)
  plotter.plot_arr(Y_pred,i,ax=ax[1,1],cmap='viridis',cb=False)
  plotter.plot_arr(m,i,ax=ax[1,2],cmap='viridis',cb=False)
  # i + step slice:
  ax[2,0].set_title(r'$log(\delta+1)$'+'\n'+f'File: {DELTA_NAME}')
  ax[2,1].set_title(f'Predicted Mask\nSlice {i+step}')
  ax[2,2].set_title(f'True Mask\nSlice {i+step}')
  plotter.plot_arr(d,i+step,ax=ax[2,0],cmap=den_cmap,logged=True)
  plotter.plot_arr(Y_pred,i+step,ax=ax[2,1],cmap='viridis',cb=False)
  plotter.plot_arr(m,i+step,ax=ax[2,2],cmap='viridis',cb=False)
  # fix axis labels to be Mpc/h:
  for axis in ax.flatten():
    plotter.set_window(0,BOXSIZE,GRID,axis,BOXSIZE)
  plt.savefig(FILE_FIG+MODEL_NAME+'-pred-comp-3x3.png',facecolor='white',bbox_inches='tight')
  print(f'Saved 3x3 comparison plot to {FILE_FIG+MODEL_NAME}-pred-comp-3x3.png')

  # use save_scores_from_fvol to save scores if we want to run the model on its own training data:
  if TRAIN_SCORE == True:
    save_scores_from_fvol(m,Y_pred,FILE_MODEL,FILE_FIG,FILE_DEN)
