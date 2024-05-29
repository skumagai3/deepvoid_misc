#!/usr/bin/env python3
import matplotlib.pyplot as plt
import imageio
import os
import io
import tensorflow as tf
import argparse
import math
from NETS_LITE import load_dataset, assemble_cube2
'''
5/29/24: model_viz.py
Visualize the filters in a 3D Convolutional layer as a GIF,
and plot activation maps for each layer in the model.

This script generates a GIF of the filters in a 3D Convolutional layer.
It will ask the user for:
0. Model name (e.g. '/Users/samkumagai/Desktop/Drexel/DeepVoid/models/TNG_D3-F32-Nm512-th0.65-sig2.4-base_L5_TL_ENC_tran_L7')
1. Directory to save the GIFs and images to. FIG_OUT


MODEL_NAME = "/Users/samkumagai/Desktop/Drexel/DeepVoid/models/TNG_D3-F32-Nm512-th0.65-sig2.4-base_L5_TL_ENC_tran_L7"
MODEL_NAME="/Users/samkumagai/Desktop/Drexel/DeepVoid/models/TNG_D2-F4-Nm128-th0.65-sig0.6-base_L0.33_FOCAL"
MODEL_NAME="/Users/samkumagai/Desktop/Drexel/DeepVoid/models/TNG_D3-F32-Nm512-th0.65-sig2.4-base_L0.33"
import os
FIG_OUT = "/Users/samkumagai/Desktop/Drexel/DeepVoid/figs/activation_GIFs/"
os.system('python3 deepvoid_misc/model_viz.py -m "{}" -o "{}"'.format(MODEL_NAME, FIG_OUT))


"/Users/samkumagai/Desktop/Drexel/DeepVoid/models/TNG_D3-F32-Nm512-th0.65-sig2.4-base_L5_TL_ENC_tran_L7"
"/Users/samkumagai/Desktop/Drexel/DeepVoid/models/TNG_D3-F32-Nm512-th0.65-sig2.4-base_L5"
"/Users/samkumagai/Library/CloudStorage/GoogleDrive-s.kumagai3@gmail.com/My Drive/models/Bolshoi_D4-F16-Nm640-th0.65-sig0.916-base_L0.122_SCCE_TL_ENC_EO_tran_L10.keras"
'''
def visualize_conv3d_filters_gif(model, fig_out):
    # Create a directory for the GIFs
    gif_dir = fig_out + model._name + '_filter_GIF'
    os.makedirs(gif_dir, exist_ok=True)
    print(f'GIFs will be saved in {gif_dir}')
    # Iterate over the layers of the model
    for layer_index, layer in enumerate(model.layers):
        # Check if the layer has the strings: decode, encode, bottleneck:
        conv_names = ['encode', 'decode', 'bottleneck', 'conv', 'block']
        if any(name in layer.name for name in conv_names):
            print(f'Layer {layer_index}: {layer.name}')
            # Get the weights of the layer
            weights, _ = layer.get_weights()
            # Normalize the weights
            weights_min = weights.min()
            weights_max = weights.max()
            weights = (weights - weights_min) / (weights_max - weights_min)
            # Get the number of filters in this layer
            num_filters = weights.shape[-1]
            # Get the depth of the filters
            depth = weights.shape[2]
            # Calculate the number of rows and columns for the grid
            grid_size = math.isqrt(num_filters)
            if grid_size * grid_size < num_filters:
                grid_size += 1
            # Create a list to store the images
            images = []
            # For each slice of the filter
            for j in range(depth):
                # Create a figure
                fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))  # Adjust the size as needed
                axs = axs.ravel()
                # For each filter
                for i in range(grid_size * grid_size):
                    if i < num_filters:
                        # Plot the slice
                        axs[i].imshow(weights[:, :, j, 0, i], cmap='gray')
                    axs[i].axis('off')
                # Save the figure to a BytesIO object
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                # Load the image from the BytesIO object and add it to the list
                images.append(imageio.imread(buf))
                # Close the figure
                plt.close(fig)
            # Save the images as a GIF
            imageio.mimsave(os.path.join(gif_dir, f'{layer.name}_filters.gif'), images)

def visualize_all_feature_maps(model, input_data):
    # DOESNT WORK YET!
    # Iterate over the layers of the model
    for layer_index, layer in enumerate(model.layers):
        # Check if the layer has the strings: decode, encode, bottleneck, conv, block:
        conv_names = ['encode', 'decode', 'bottleneck', 'conv', 'block']
        if any(name in layer.name for name in conv_names):
            print(f'Layer {layer_index}: {layer.name}')
            # Create a Model that outputs the feature maps of the current layer
            feature_map_model = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
            # Run the input data through this Model
            feature_maps = feature_map_model.predict(input_data)
            # reassemble subcubes:
            feature_maps = assemble_cube2(feature_maps, 512, 128, 64)
            # Get the number of feature maps
            num_feature_maps = feature_maps.shape[-1]
            # Calculate the number of rows and columns for the grid
            grid_size = math.isqrt(num_feature_maps)
            if grid_size * grid_size < num_feature_maps:
                grid_size += 1
            # Create a figure
            fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))  # Adjust the size as needed
            axs = axs.ravel()
            # For each feature map
            for i in range(grid_size * grid_size):
                if i < num_feature_maps:
                    # Plot the feature map
                    axs[i].imshow(feature_maps[0, :, :, i], cmap='gray')
                axs[i].axis('off')
            # Show the figure
            plt.show()

if __name__ == '__main__':
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Generate a GIF of the filters in a 3D Convolutional layer')
    parser.add_argument('-m', '--model', type=str, help='Model path', required=True)
    parser.add_argument('-o', '--fig_out', type=str, help='Directory to save the GIFs', required=True)
    parser.add_argument('-i', '--input', type=str, help='Input data path. FILE_DEN', required=True)
    args = parser.parse_args()
    model = tf.keras.models.load_model(args.model,compile=False)
    print('Model loaded successfully!')
    #print('Model summary:')
    #model.summary()
    # Visualize the filters as a grid of GIFs:
    visualize_conv3d_filters_gif(model, args.fig_out)
    
    # Load input data for activation maps:
    #FILE_DEN = args.input
    #X_data = load_dataset(FILE_DEN,128,64)
    # Visualize the activation maps for each layer:
    #visualize_all_feature_maps(model, X_data)

        
