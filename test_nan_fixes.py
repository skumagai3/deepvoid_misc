#!/usr/bin/env python3
"""
Test script to verify NaN fixes in preprocessing methods
"""
import numpy as np
import NETS_LITE as nets

def test_preprocessing_edge_cases():
    """Test preprocessing methods with edge cases that could cause NaNs"""
    
    print("Testing preprocessing methods for NaN resistance...")
    
    # Test case 1: All values are the same (constant array)
    print("\n1. Testing constant array...")
    constant_array = np.full((10, 10, 10), 5.0)
    
    # Test minmax
    result = nets.minmax(constant_array)
    print(f"   minmax result has NaNs: {np.any(np.isnan(result))}")
    print(f"   minmax result unique values: {np.unique(result)}")
    
    # Test standardize  
    result = nets.standardize(constant_array)
    print(f"   standardize result has NaNs: {np.any(np.isnan(result))}")
    print(f"   standardize result unique values: {np.unique(result)}")
    
    # Test case 2: Array with very small variance after clipping
    print("\n2. Testing array with small variance after clipping...")
    small_var_array = np.random.normal(1.0, 0.001, (10, 10, 10))
    # Add a few outliers that will be clipped
    small_var_array[0, 0, 0] = 1000.0
    small_var_array[0, 0, 1] = -1000.0
    
    # Simulate robust preprocessing
    clipped = np.clip(small_var_array, np.percentile(small_var_array, 1), np.percentile(small_var_array, 99))
    std_val = np.std(clipped)
    print(f"   Standard deviation after clipping: {std_val}")
    
    if std_val == 0:
        result = np.zeros_like(clipped)
    else:
        result = (clipped - np.median(clipped)) / std_val
    
    print(f"   Robust preprocessing result has NaNs: {np.any(np.isnan(result))}")
    
    # Test case 3: Log transform of array with zeros/negatives
    print("\n3. Testing log transform with edge values...")
    edge_array = np.array([0, 0, 0, 1e-10, 1e-6, 1.0, 10.0]).reshape((7, 1, 1))
    
    log_array = np.log10(edge_array + 1e-6)
    std_val = np.std(log_array)
    print(f"   Standard deviation after log transform: {std_val}")
    
    if std_val == 0:
        result = np.zeros_like(log_array)
    else:
        result = (log_array - np.mean(log_array)) / std_val
        
    print(f"   Log transform result has NaNs: {np.any(np.isnan(result))}")
    
    print("\n✅ All edge case tests completed!")

if __name__ == "__main__":
    test_preprocessing_edge_cases()
