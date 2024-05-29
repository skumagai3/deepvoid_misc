#!/usr/bin/env python3
'''
5/28/24: This script interactively generates a shell script file.
It will ask the user for:
0. Picotte or Colab?

1. Training or Transfer Learning?

# if Training:
2. Simulation (TNG/BOL)
3. Interparticle spacing (Lambda)
4. Depth of the network
5. Number of filters in the first layer
6. Loss function (CCE, SCCE, FOCAL_CCE)
7. GRID size (512 for TNG, 640 for BOL)
8. Batch normalization (True/False)
9. Dropout rate (0.0 to 1.0)
10. Multiple GPUs (True/False)
11. Number of epochs
12. Initial learning rate
13. TensorBoard (True/False)

# if Transfer Learning:
Same as above, but with the addition of:
14. Transfer Lambda
15. TL Type (ENC, ENC_EO, LL)
'''
import os
import sys
import argparse
import datetime
import numpy as np

def main():
    # set preamble:
    preamble = '#!/bin/bash\n'

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Generate a shell script for training or transfer learning')
    parser.add_argument('-o', '--output', type=str, help='Output shell script file', required=True)
    args = parser.parse_args()

    # Get the output file
    output_file = args.output

    # Open the output file
    with open(output_file, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# Shell script generated on {}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        f.write('\n')
        f.write('python3 train.py \\\n')
        f.write('    --simulation TNG \\\n')
        f.write('    --lambda 0.7 \\\n')
        f.write('    --depth 3 \\\n')
        f.write('    --filters 16 \\\n')
        f.write('    --loss SCCE \\\n')
        f.write('    --grid 512 \\\n')
        f.write('    --batch_norm True \\\n')
        f.write('    --dropout 0.2 \\\n')
        f.write('    --multi_gpu False \\\n')
        f.write('    --epochs 100 \\\n')
        f.write('    --lr 0.001 \\\n')
        f.write('    --tensorboard True\n')

    print('Shell script generated in {}'.format(output_file))