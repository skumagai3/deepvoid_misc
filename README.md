# DeepVoid: Deep Learning for Cosmic Void Detection

A collection of deep learning tools for identifying and analyzing cosmic voids in large-scale structure simulations using 3D U-Net architectures.

## Overview

DeepVoid uses deep convolutional neural networks to detect cosmic voids in dark matter density fields from cosmological simulations. The morphology of large-scale structure can be defined in many different ways. The tidal tensor approach is one good physics-based dynamical definition that can be easily derived from simulations. In this scheme, we count how many eigenvalues of the tidal tensor (the Hessian of the gravitational potential) exceed a given threshold to determine morphology: void (no eigenvalues exceed threshold), sheet (one eigenvalue exceeds threshold), filament (two eigenvalues exceed threshold), or node/halo (all three eigenvalues exceed threshold).

The basic tidal tensor scheme identifies the most underdense centers of voids. In this implementation, we use an eigenvalue threshold of 0.65 (visible in the mask filenames) to define void regions more comprehensively. However, this architecture is flexible and can be adapted to different physical void definitions beyond the tidal tensor approach.

The goal is to train models on simulation data that can then take galaxy surveys as input and output LSS morphology classifications. For void finding specifically, this can be used to create void catalogues that are then stacked (averaged) to create samples for cosmological tests like Alcock-Paczyński tests, integrated Sachs-Wolfe studies, and other analyses. It could also be interesting to study how local morphology influences galaxy properties.

The immediate target is applying these techniques to observational galaxy surveys such as DESI and other high-redshift surveys.

The project includes training scripts, prediction tools, and comprehensive documentation for working with IllustrisTNG and Bolshoi simulations.

## Key Features

- **3D U-Net Architecture**: Specialized for volumetric cosmic void detection
- **Attention Mechanisms**: Enhanced feature extraction with attention gates
- **Lambda Conditioning**: Scale-aware training for multi-resolution analysis
- **Curricular Training**: Progressive multi-scale learning approach
- **Transfer Learning**: Efficient adaptation between different scales
- **Improved Loss Functions**: Addresses void/wall prediction bias
- **Redshift Space Distortions**: Realistic observational effects simulation

## Quick Start

### Basic Training
```bash
python DV_MULTI_TRAIN.py /path/to/data/ TNG 0.33 4 16 SCCE_Class_Penalty_Fixed 512 \
    --ATTENTION_UNET --LAMBDA_CONDITIONING --BATCH_SIZE 8
```

### Curricular Training (Recommended)
```bash
python curricular.py /path/to/data/ 4 16 SCCE_Class_Penalty_Fixed \
    --USE_ATTENTION --LAMBDA_CONDITIONING --BATCH_SIZE 8
```

## Core Components

### Training Scripts
- **`DV_MULTI_TRAIN.py`**: Primary training script for 3D U-Net models
- **`DV_MULTI_TRANSFER.py`**: Transfer learning between different scales
- **`attention_test.py`**: Attention U-Net experimentation
- **`curricular.py`**: Multi-scale progressive training

### Core Libraries
- **`NETS_LITE.py`**: Neural network architectures and loss functions
- **`plotter.py`**: Visualization utilities
- **`volumes.py`**: Binary volume file I/O functions

### Demo and Analysis
- **`DeepVoid_demo.ipynb`**: Complete tutorial with TNG300 example
- **`diagnose_bias.py`**: Tool for analyzing prediction bias

## Progressive Curricular Learning

DeepVoid implements a sophisticated curricular learning approach that progressively trains models across multiple spatial scales, mimicking how cosmic structures form hierarchically in the universe.

### Methodology

The curricular training approach addresses the challenge of multi-scale void detection by:

1. **Hierarchical Scale Training**: Starting with the largest observable scale (lowest resolution) and progressively moving to smaller scales (higher resolution)
2. **Knowledge Transfer**: Each stage builds upon knowledge learned from the previous scale, improving convergence and final performance
3. **Scale-Adaptive Architecture**: The same network architecture adapts to different resolutions through lambda conditioning

### Key Benefits

- **Improved Convergence**: Progressive training typically converges faster than single-scale training
- **Better Feature Learning**: Models learn robust features that work across multiple scales
- **Physical Consistency**: Training order matches the hierarchical formation of cosmic structure
- **Reduced Overfitting**: Progressive complexity reduces the risk of overfitting at high resolutions

### Implementation

The `curricular.py` script automates this process:

```bash
# Progressive training across scales: 0.33 → 0.25 → 0.17 Mpc/h
python curricular.py /path/to/data/ 4 16 SCCE_Class_Penalty_Fixed \
    --USE_ATTENTION --LAMBDA_CONDITIONING --BATCH_SIZE 8
```

This approach trains models sequentially at separations of 0.33, 0.25, and 0.17 Mpc/h, with each stage initializing from the previous stage's trained weights.

## Data Structure

The training scripts expect the following directory structure:

```
ROOT_DIR/
├── data/
│   ├── TNG/
│   │   ├── DM_DEN_snap99_Nm=512.fvol          # Full DM density field
│   │   └── subs1_mass_Nm512_L{L}_d_None_smooth.fvol  # Subhalo data
│   └── Bolshoi/
│       ├── Bolshoi_halo_CIC_640_L=0.122.fvol  # Full DM density field
│       └── Bolshoi_halo_CIC_640_L={L}.fvol    # Halo data
├── models/      # Trained model outputs
└── preds/       # Prediction outputs
```

## Loss Functions

DeepVoid includes several loss functions optimized for cosmic void detection:

1. **`SCCE_Class_Penalty_Fixed`** (Recommended) - Balanced approach addressing void/wall bias
2. **`SCCE_Proportion_Aware`** - Maintains target class proportions
3. **`SCCE_Balanced_Class_Penalty`** - Alternative balanced approach
4. **`SCCE`** - Standard sparse categorical crossentropy

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Getting Started Guide](docs/STANDARD_SCRIPTS_USAGE_GUIDE.md)** - Complete usage examples
- **[Curricular Training](docs/CURRICULAR_USAGE_GUIDE.md)** - Progressive training approach
- **[Loss Function Improvements](docs/LOSS_FUNCTION_IMPROVEMENTS.md)** - Technical details on bias fixes

## Requirements

Install dependencies using:
```bash
pip install -r requirements.txt
```

Key dependencies include:
- TensorFlow >= 2.15
- NumPy
- SciPy
- Matplotlib
- scikit-learn

## Usage Examples

### Standard Training
```bash
# Basic U-Net training
python DV_MULTI_TRAIN.py /data TNG 0.33 4 16 SCCE 512

# With attention and lambda conditioning
python DV_MULTI_TRAIN.py /data TNG 0.33 4 16 SCCE_Class_Penalty_Fixed 512 \
    --ATTENTION_UNET --LAMBDA_CONDITIONING
```

### Transfer Learning
```bash
# Transfer from one scale to another
python DV_MULTI_TRANSFER.py /data model_name density_file TL_TYPE TNG 512 \
    --LOSS SCCE_Class_Penalty_Fixed
```

## Citation

If you use DeepVoid in your research, please cite:

```bibtex
@ARTICLE{2025arXiv250421134K,
       author = {{Kumagai}, Sam and {Vogeley}, Michael S. and {Aragon-Calvo}, Miguel A. and {Douglass}, Kelly A. and {BenZvi}, Segev and {Neyrinck}, Mark},
        title = "{DeepVoid: A Deep Learning Void Detector}",
      journal = {arXiv e-prints},
     keywords = {Instrumentation and Methods for Astrophysics},
         year = 2025,
        month = apr,
          eid = {arXiv:2504.21134},
        pages = {arXiv:2504.21134},
          doi = {10.48550/arXiv.2504.21134},
archivePrefix = {arXiv},
       eprint = {2504.21134},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250421134K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Contact

For questions or collaboration opportunities, please contact [contact information].

---

**Note**: This is research software under active development. Some features may be experimental.
