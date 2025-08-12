# File Naming Strategy for curricular_pred.py

## Overview
The `curricular_pred.py` script now ensures that all output files include both the L_PRED (interparticle separation) value and the preprocessing method in their filenames. This prevents file overwrites when running predictions multiple times with different L_VAL values or preprocessing methods.

## File Naming Patterns

### Prediction Files (PRED_PATH)
- **Main predictions**: `{MODEL_NAME}_predictions_L{L_PRED}_{PREPROCESSING}.npy`
- **Slice predictions**: `{MODEL_NAME}_slice_predictions_L{L_PRED}_{PREPROCESSING}.npy`
- **Legacy .fvol format**: `{MODEL_NAME}_predictions_L{L_PRED}_{PREPROCESSING}.fvol`

### Figure Files (FIG_PATH/{MODEL_NAME}/)
- **Confusion matrix**: `{MODEL_NAME}_confusion_matrix_L{L_PRED}_{PREPROCESSING}.npy`
- **Slice plot data**: `{MODEL_NAME}_slice_data_L{L_PRED}_{PREPROCESSING}.npz`
- **ROC curves**: `{MODEL_NAME}_temp_L{L_PRED}_{PREPROCESSING}_ROC_OvR.png`
- **PR curves**: `{MODEL_NAME}_temp_L{L_PRED}_{PREPROCESSING}_PR.png`
- **Confusion matrix plot**: `{MODEL_NAME}_temp_L{L_PRED}_{PREPROCESSING}_cm.png`
- **Prediction comparison**: `{MODEL_NAME}_temp_L{L_PRED}_{PREPROCESSING}-pred-comp.png`

### Temporary Files (MODEL_PATH)
- **Temporary model**: `{MODEL_NAME}_temp_L{L_PRED}_{PREPROCESSING}.keras`

## Example Filenames

For a model named "TNG_curricular_model" with L_PRED=10 and PREPROCESSING=robust:

### Prediction Files
```
TNG_curricular_model_predictions_L10_robust.npy
TNG_curricular_model_slice_predictions_L10_robust.npy
TNG_curricular_model_predictions_L10_robust.fvol
```

### Figure Files
```
TNG_curricular_model_confusion_matrix_L10_robust.npy
TNG_curricular_model_slice_data_L10_robust.npz
TNG_curricular_model_temp_L10_robust_ROC_OvR.png
TNG_curricular_model_temp_L10_robust_PR.png
TNG_curricular_model_temp_L10_robust_cm.png
TNG_curricular_model_temp_L10_robust-pred-comp.png
```

## Benefits

1. **No File Conflicts**: Running predictions with different L_VAL values won't overwrite each other
2. **Preprocessing Differentiation**: Running the same model with different preprocessing methods creates separate files
3. **Easy Organization**: Files are clearly labeled with their parameters
4. **Batch Processing Safe**: You can run multiple predictions simultaneously without conflicts

## Usage Examples

### Different L_VAL values (same preprocessing)
```bash
python curricular_pred.py /path/to/project/ model_name 3 --PREPROCESSING robust
python curricular_pred.py /path/to/project/ model_name 5 --PREPROCESSING robust  
python curricular_pred.py /path/to/project/ model_name 10 --PREPROCESSING robust
```
Files: `model_name_predictions_L3_robust.npy`, `model_name_predictions_L5_robust.npy`, `model_name_predictions_L10_robust.npy`

### Same L_VAL, different preprocessing
```bash
python curricular_pred.py /path/to/project/ model_name 10 --PREPROCESSING standard
python curricular_pred.py /path/to/project/ model_name 10 --PREPROCESSING robust
python curricular_pred.py /path/to/project/ model_name 10 --PREPROCESSING log_transform
```
Files: `model_name_predictions_L10_standard.npy`, `model_name_predictions_L10_robust.npy`, `model_name_predictions_L10_log_transform.npy`

## File Cleanup
- Temporary model files are automatically cleaned up after each run
- All other files are preserved for analysis
- Use the L_PRED and PREPROCESSING values in filenames to identify which files belong to which run
