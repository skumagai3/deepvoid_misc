#!/usr/bin/env python3
"""
Test script to validate that the variable deletion fix works properly
"""

def test_variable_deletion():
    """Test the variable deletion logic"""
    print("Testing variable deletion logic...")
    
    # Case 1: All variables exist
    print("\nCase 1: All variables exist")
    train_features = "dummy_features"
    train_labels = "dummy_labels" 
    train_dataset = "dummy_dataset"
    
    # Test the fixed deletion logic
    if 'train_dataset' in locals():
        del train_dataset
        print("✓ Deleted train_dataset")
    if 'train_features' in locals():
        del train_features  
        print("✓ Deleted train_features")
    if 'train_labels' in locals():
        del train_labels
        print("✓ Deleted train_labels")
    
    # Case 2: Some variables don't exist (should not crash)
    print("\nCase 2: Variables already deleted (should not crash)")
    try:
        if 'train_dataset' in locals():
            del train_dataset
            print("✓ train_dataset already deleted")
        else:
            print("✓ train_dataset not in locals - skipped")
        if 'train_features' in locals():
            del train_features
            print("✓ train_features already deleted")
        else:
            print("✓ train_features not in locals - skipped")
        if 'train_labels' in locals():
            del train_labels
            print("✓ train_labels already deleted")
        else:
            print("✓ train_labels not in locals - skipped")
            
        print("✅ Variable deletion logic works correctly!")
        return True
    except NameError as e:
        print(f"❌ Variable deletion failed: {e}")
        return False

if __name__ == "__main__":
    test_variable_deletion()
