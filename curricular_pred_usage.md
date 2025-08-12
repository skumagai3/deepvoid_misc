# Curricular Prediction Script Usage

## Overview
The `curricular_pred.py` script now supports preprocessing arguments to ensure consistent evaluation of models trained with different preprocessing methods.

## Basic Usage
```bash
python curricular_pred.py ROOT_DIR MODEL_NAME L_PRED --PREPROCESSING method
```

## Preprocessing Options
- `standard`: Min-max normalization to [0,1] (default)
- `robust`: Outlier clipping + capping to [-3,3]
- `log_transform`: Log10 transformation + standardization
- `clip_extreme`: Conservative outlier handling

## Examples

### Standard Preprocessing (Default)
```bash
python curricular_pred.py /path/to/project/ TNG_curricular_model 10 --PREPROCESSING standard
```

### Robust Preprocessing
```bash
python curricular_pred.py /path/to/project/ TNG_curricular_model 10 --PREPROCESSING robust
```

### Log Transform Preprocessing
```bash
python curricular_pred.py /path/to/project/ TNG_curricular_model 10 --PREPROCESSING log_transform
```

### Clip Extreme Preprocessing
```bash
python curricular_pred.py /path/to/project/ TNG_curricular_model 10 --PREPROCESSING clip_extreme
```

## Important Notes
- **The preprocessing method used for prediction MUST match the preprocessing method used during training**
- Models trained with different preprocessing methods need to be evaluated with their respective preprocessing
- The default is 'standard' preprocessing for backward compatibility
- Check your training logs or configuration to determine which preprocessing was used

## Additional Options
You can combine preprocessing with other options:

```bash
python curricular_pred.py /path/to/project/ TNG_curricular_model 10 \
  --PREPROCESSING robust \
  --BATCH_SIZE 4 \
  --SAVE_PREDICTIONS \
  --TEST_MODE
```

## Troubleshooting
- If predictions seem inconsistent, verify the preprocessing method matches training
- Use `--TEST_MODE` for quick testing with small data samples
- Check memory usage with smaller `--BATCH_SIZE` if needed
