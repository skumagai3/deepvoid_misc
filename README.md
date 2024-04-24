# deepvoid_misc
Collection of training scripts and other assorted files. 

Files:
- DeepVoid_demo.ipynb: jupyter notebook with an example of DeepVoid running on a low-res realization of TNG300-3-Dark.
- NETS_LITE.py: python script containing functions for loading data, etc.
- DV_MULTI_TRAIN.py: python script for training a 3D U-net.
- plotter.py: python script containing utility plotting functions.
- volumes.py: python script that has functions to read in binary float volumes (.fvols)
- requirements.txt: use a venv and pip install -r to install these. Should work with code (last checked April 12, 2024)

DV_MULTI_TRAIN.py:
Usage: python3 DV_MULTI_TRAIN.py <ROOT_DIR> <SIM> <L> <DEPTH> <FILTERS> <UNIFORM_FLAG> <BATCHNORM> <DROPOUT> <LOSS>, 
        where ROOT_DIR is your root directory where models, predictions, figures will be saved,
        SIM is BOL or TNG, L is the interparticle separation in Mpc/h,
        DEPTH is the depth of the U-Net, FILTERS is the number of filters in the first layer,
        and UNIFORM_FLAG is 1 if you want to use identical masses for all subhaloes, 0 if not.
        BATCHNORM is 1 if you want to use batch normalization, 0 if not.
        DROPOUT is the dropout rate, and LOSS is the loss function to use.
        LOSS is one of 'CCE', 'FOCAL_CCE', 'DICE_AVG', or 'DICE_VOID'.
(NOTE: only CCE and FOCAL_CCE are implemented.)


NOTE also that DV_MULTI_TRAIN expects the file structure to be as such:
ROOT_DIR
|-- data
|   |-- TNG
|   |   |-- DM_DEN_snap99_Nm=512.fvol (full DM density)
|   |   `-- subs1_mass_Nm512_L{L}_d_None_smooth.fvol
|   `-- Bolshoi
|       |-- 'Bolshoi_halo_CIC_640_L=0.122.fvol' (full DM density)
|       `-- 'Bolshoi_halo_CIC_640_L={L}.fvol'
|-- models
`-- preds
