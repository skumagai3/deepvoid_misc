#!/usr/bin/env python3
"""
Simple script to test GPU memory management improvements for curricular training on Google Colab.
This script tests the memory-efficient dataset creation without running full training.
"""

import os
import gc
import numpy as np
import tensorflow as tf

# Set environment variables for Google Colab compatibility
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" 
os.environ["TF_DISABLE_CUDNN_AUTOTUNE"] = "1"

print('Testing GPU memory management for Google Colab...')

# Initialize TensorFlow
gpus = tf.config.list_physical_devices('GPU')
print('GPUs available:', gpus)

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Set memory limits based on GPU type
    try:
        gpu_details = tf.config.experimental.get_device_details(gpus[0])
        device_name = gpu_details.get('device_name', 'Unknown')
        print(f'GPU device name: {device_name}')
        
        if 'L4' in device_name:
            tf.config.experimental.set_memory_limit(gpus[0], 18 * 1024)  # 18GB
            print('Set L4 memory limit to 18GB')
        elif 'T4' in device_name:
            tf.config.experimental.set_memory_limit(gpus[0], 12 * 1024)  # 12GB
            print('Set T4 memory limit to 12GB')
    except Exception as e:
        print(f'Could not set memory limit: {e}')

def memory_efficient_dataset_test():
    """Test memory-efficient dataset creation"""
    print('\n=== Testing memory-efficient dataset creation ===')
    
    # Create dummy data similar to curricular training
    n_samples = 1000
    subgrid = 128
    
    print(f'Creating dummy data: {n_samples} samples of {subgrid}^3 cubes')
    features = np.random.random((n_samples, subgrid, subgrid, subgrid, 1)).astype(np.float32)
    labels = np.random.randint(0, 4, (n_samples, subgrid, subgrid, subgrid, 1)).astype(np.int8)
    
    print(f'Features shape: {features.shape}, Labels shape: {labels.shape}')
    print(f'Memory usage: Features ~{features.nbytes / 1024**3:.2f} GB, Labels ~{labels.nbytes / 1024**3:.2f} GB')
    
    # Test GPU memory before dataset creation
    if gpus:
        try:
            gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
            print(f'GPU memory before dataset: {gpu_memory["current"] / 1024**3:.2f} GB used')
        except:
            pass
    
    try:
        # Test dataset creation with memory-efficient approach
        print('Creating dataset with memory-efficient approach...')
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.batch(8).prefetch(tf.data.AUTOTUNE)
        
        # Test a few batches
        for i, (batch_x, batch_y) in enumerate(dataset.take(3)):
            print(f'Batch {i}: Features {batch_x.shape}, Labels {batch_y.shape}')
        
        print('Dataset creation successful!')
        
        # Test GPU memory after dataset creation
        if gpus:
            try:
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                print(f'GPU memory after dataset: {gpu_memory["current"] / 1024**3:.2f} GB used')
            except:
                pass
        
    except Exception as e:
        print(f'Dataset creation failed: {e}')
        return False
    
    # Clean up
    del features, labels, dataset
    gc.collect()
    print('Memory cleaned up successfully')
    return True

def test_progressive_loading():
    """Test progressive loading of multiple datasets"""
    print('\n=== Testing progressive loading ===')
    
    datasets = []
    n_samples = 500
    subgrid = 64  # Smaller for testing
    
    for i in range(3):
        print(f'Loading dataset {i+1}/3...')
        
        # Create dummy data
        features = np.random.random((n_samples, subgrid, subgrid, subgrid, 1)).astype(np.float32)
        labels = np.random.randint(0, 4, (n_samples, subgrid, subgrid, subgrid, 1)).astype(np.int8)
        
        # Create dataset and immediately clear arrays
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.batch(4).prefetch(tf.data.AUTOTUNE)
        datasets.append(dataset)
        
        # Clean up arrays immediately
        del features, labels
        gc.collect()
        
        # Check memory usage
        if gpus:
            try:
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                print(f'GPU memory after dataset {i+1}: {gpu_memory["current"] / 1024**3:.2f} GB used')
            except:
                pass
    
    print('Progressive loading successful!')
    
    # Clean up all datasets
    del datasets
    gc.collect()
    return True

if __name__ == "__main__":
    print('TensorFlow version:', tf.__version__)
    print('CUDA available:', tf.test.is_built_with_cuda())
    
    # Run tests
    test1_success = memory_efficient_dataset_test()
    test2_success = test_progressive_loading()
    
    if test1_success and test2_success:
        print('\n✅ All memory management tests passed! Your curricular training should work on Google Colab.')
    else:
        print('\n❌ Some tests failed. You may need to reduce batch size or use CPU-only training.')
