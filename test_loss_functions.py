#!/usr/bin/env python3
"""
Quick test script to verify the new loss functions work correctly.
"""

import sys
sys.path.append('.')
import NETS_LITE as nets
import tensorflow as tf
import numpy as np

def test_loss_functions():
    """Test that all loss functions can be called without errors."""
    print("Testing improved loss functions in NETS_LITE.py...")
    
    # Create dummy data
    batch_size = 2
    spatial_dims = (8, 8, 8)
    n_classes = 4
    
    # Create dummy true labels (sparse format)
    y_true = np.random.randint(0, n_classes, size=(batch_size,) + spatial_dims + (1,))
    y_true = tf.constant(y_true, dtype=tf.int64)
    
    # Create dummy predictions (softmax probabilities)
    y_pred_logits = np.random.randn(batch_size, *spatial_dims, n_classes)
    y_pred = tf.nn.softmax(y_pred_logits, axis=-1)
    
    print(f"Input shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")
    
    # Test original functions
    print("\n1. Testing original SCCE_Class_Penalty...")
    try:
        loss1 = nets.SCCE_Class_Penalty(y_true, y_pred)
        print(f"   ✓ SCCE_Class_Penalty: {tf.reduce_mean(loss1):.4f}")
    except Exception as e:
        print(f"   ✗ SCCE_Class_Penalty failed: {e}")
    
    print("\n2. Testing SCCE_Balanced_Class_Penalty...")
    try:
        loss2 = nets.SCCE_Balanced_Class_Penalty(y_true, y_pred)
        print(f"   ✓ SCCE_Balanced_Class_Penalty: {tf.reduce_mean(loss2):.4f}")
    except Exception as e:
        print(f"   ✗ SCCE_Balanced_Class_Penalty failed: {e}")
    
    # Test new improved functions
    print("\n3. Testing SCCE_Class_Penalty_Fixed...")
    try:
        loss3 = nets.SCCE_Class_Penalty_Fixed(y_true, y_pred)
        print(f"   ✓ SCCE_Class_Penalty_Fixed: {tf.reduce_mean(loss3):.4f}")
    except Exception as e:
        print(f"   ✗ SCCE_Class_Penalty_Fixed failed: {e}")
    
    print("\n4. Testing SCCE_Proportion_Aware...")
    try:
        loss4 = nets.SCCE_Proportion_Aware(y_true, y_pred)
        print(f"   ✓ SCCE_Proportion_Aware: {tf.reduce_mean(loss4):.4f}")
    except Exception as e:
        print(f"   ✗ SCCE_Proportion_Aware failed: {e}")
    
    print("\n✓ All loss function tests completed!")
    print("\nAvailable loss functions for curricular training:")
    print("  - SCCE_Class_Penalty (original, with balanced params)")
    print("  - SCCE_Balanced_Class_Penalty")  
    print("  - SCCE_Class_Penalty_Fixed")
    print("  - SCCE_Proportion_Aware")
    print("\nRecommended for your void/wall bias issue:")
    print("  python curricular.py /path/to/data 4 16 SCCE_Class_Penalty_Fixed")

if __name__ == "__main__":
    test_loss_functions()
