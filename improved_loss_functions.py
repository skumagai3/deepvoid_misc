#!/usr/bin/env python3
"""
Fixed SCCE_Class_Penalty loss function to address void/wall prediction bias
"""

import tensorflow as tf

def SCCE_Class_Penalty_Fixed(y_true, y_pred, void_penalty=2.0, wall_penalty=1.0, minority_boost=2.0):
    """
    Balanced SCCE loss that encourages correct class prediction without heavy bias.
    
    Args:
        y_true: true labels (sparse format)  
        y_pred: predicted probabilities (softmax output)
        void_penalty: penalty factor for predicting void when it's not void (reduced from 8.0)
        wall_penalty: penalty factor for predicting wall when it's not wall
        minority_boost: boost factor for correctly predicting minority classes
    """
    # Base SCCE loss
    scce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    
    # Get class predictions and true labels
    y_pred_classes = tf.argmax(y_pred, axis=-1)
    y_true_classes = tf.cast(tf.squeeze(y_true, axis=-1), tf.int64)
    
    # Balanced penalties for all major classes
    false_void_mask = tf.logical_and(y_pred_classes == 0, y_true_classes != 0)
    false_wall_mask = tf.logical_and(y_pred_classes == 1, y_true_classes != 1)
    
    false_void_penalty = tf.reduce_mean(tf.cast(false_void_mask, tf.float32)) * void_penalty
    false_wall_penalty = tf.reduce_mean(tf.cast(false_wall_mask, tf.float32)) * wall_penalty
    
    # Rewards for correct predictions (but balanced)
    void_correct = tf.logical_and(y_pred_classes == 0, y_true_classes == 0)
    wall_correct = tf.logical_and(y_pred_classes == 1, y_true_classes == 1)
    filament_correct = tf.logical_and(y_pred_classes == 2, y_true_classes == 2)
    halo_correct = tf.logical_and(y_pred_classes == 3, y_true_classes == 3)
    
    # Give modest rewards for correct predictions, higher rewards for minority classes
    void_reward = -tf.reduce_mean(tf.cast(void_correct, tf.float32)) * 0.1  # Small reward for dominant class
    wall_reward = -tf.reduce_mean(tf.cast(wall_correct, tf.float32)) * 0.2  # Slightly higher for 2nd class
    filament_reward = -tf.reduce_mean(tf.cast(filament_correct, tf.float32)) * minority_boost
    halo_reward = -tf.reduce_mean(tf.cast(halo_correct, tf.float32)) * minority_boost * 2.0
    
    # Penalties for missing minority classes
    missed_filament = tf.logical_and(y_pred_classes != 2, y_true_classes == 2)
    missed_halo = tf.logical_and(y_pred_classes != 3, y_true_classes == 3)
    
    missed_filament_penalty = tf.reduce_mean(tf.cast(missed_filament, tf.float32)) * minority_boost
    missed_halo_penalty = tf.reduce_mean(tf.cast(missed_halo, tf.float32)) * minority_boost * 1.5
    
    # Combine all penalties and rewards
    total_penalty = (false_void_penalty + false_wall_penalty +
                     missed_filament_penalty + missed_halo_penalty +
                     void_reward + wall_reward + filament_reward + halo_reward)
    
    return scce_loss + total_penalty

def SCCE_Proportion_Aware(y_true, y_pred, target_props=[0.65, 0.25, 0.08, 0.02], prop_weight=1.0):
    """
    SCCE loss that encourages the model to match target class proportions.
    
    Args:
        y_true: true labels (sparse format)
        y_pred: predicted probabilities (softmax output)  
        target_props: target proportions for [void, wall, filament, halo]
        prop_weight: weight for the proportion penalty
    """
    # Base SCCE loss
    scce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    
    # Calculate predicted proportions
    pred_probs = tf.nn.softmax(y_pred, axis=-1)
    pred_props = tf.reduce_mean(pred_probs, axis=[0, 1, 2, 3])  # Average over all spatial dimensions
    
    # Calculate proportion penalty
    target_props_tensor = tf.constant(target_props, dtype=tf.float32)
    prop_diff = tf.square(pred_props - target_props_tensor)
    prop_penalty = tf.reduce_sum(prop_diff) * prop_weight
    
    return scce_loss + prop_penalty
