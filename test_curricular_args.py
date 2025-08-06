#!/usr/bin/env python3
"""
Quick test script to verify curricular.py argument parsing works correctly
"""

import subprocess
import sys

def test_help_message():
    """Test that help message shows new arguments"""
    try:
        result = subprocess.run([sys.executable, 'curricular.py', '--help'], 
                              capture_output=True, text=True, timeout=10)
        
        help_output = result.stdout
        
        # Check for new arguments
        new_args = ['--PREPROCESSING', '--WARMUP_EPOCHS', '--VALIDATION_STRATEGY', '--TARGET_LAMBDA']
        
        print("Testing argument parsing...")
        for arg in new_args:
            if arg in help_output:
                print(f"✓ {arg} found in help message")
            else:
                print(f"✗ {arg} NOT found in help message")
        
        # Check for preprocessing choices
        if 'robust' in help_output and 'log_transform' in help_output:
            print("✓ Preprocessing choices found")
        else:
            print("✗ Preprocessing choices missing")
            
        # Check for validation strategy choices
        if 'target' in help_output and 'stage' in help_output and 'hybrid' in help_output:
            print("✓ Validation strategy choices found")
        else:
            print("✗ Validation strategy choices missing")
            
        print("\nFirst 20 lines of help output:")
        print('\n'.join(help_output.split('\n')[:20]))
        
    except subprocess.TimeoutExpired:
        print("Help message test timed out - likely import issues")
    except Exception as e:
        print(f"Error testing help message: {e}")

if __name__ == "__main__":
    test_help_message()
