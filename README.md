# deepvoid_misc
Collection of training scripts and other assorted files. 

Files:
- NETS_LITE.py: python script containing functions for loading data, etc.
- DV_MULTI_TRAIN.py: python script for training a 3D U-net.
- plotter.py: python script containing utility plotting functions.
- volumes.py: python script that has functions to read in binary float volumes (.fvols)
- requirements.txt: use a venv and pip install -r to install these. Should work with code (last checked April 12, 2024)

Usage: python3 dv-train-nonbinary.py SIM L DEPTH FILTERS UNIFORM_FLAG BATCHNORM DROPOUT LOSS, 
        where SIM is either TNG or BOL, L is the interparticle separation in Mpc/h,
        DEPTH is the depth of the U-Net, FILTERS is the number of filters in the first layer,
        and UNIFORM_FLAG is 1 if you want to use identical masses for all subhaloes, 0 if not.
        BATCHNORM is 1 if you want to use batch normalization, 0 if not.
        DROPOUT is the dropout rate, and LOSS is the loss function to use.
        LOSS is one of 'CCE', 'FOCAL_CCE', 'DICE_AVG', or 'DICE_VOID'

(NOTE: only CCE and FOCAL_CCE are implemented.)
