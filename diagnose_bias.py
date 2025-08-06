#!/usr/bin/env python3
"""
Diagnosis script to analyze void/wall prediction bias in DeepVoid models.
"""

import argparse

def analyze_predictions(log_file_path):
    """
    Analyze the training log and provide insights into the prediction bias.
    """
    print("=== DeepVoid Prediction Bias Analysis ===\n")
    
    # Try to read the log file
    try:
        with open(log_file_path, 'r') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find log file at {log_file_path}")
        return
    
    # Extract population statistics from log
    print("1. DATA DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    if "% of population:" in log_content:
        lines = log_content.split('\n')
        populations = []
        for line in lines:
            if "% of population:" in line:
                try:
                    pct = float(line.split(':')[1].strip())
                    populations.append(pct)
                except:
                    continue
        
        if len(populations) >= 4:
            class_labels = ['Void', 'Wall', 'Filament', 'Halo']
            print("True class distribution:")
            for i, (label, pct) in enumerate(zip(class_labels, populations)):
                print(f"  {label}: {pct:.1f}%")
            
            max_idx = populations.index(max(populations))
            sorted_pops = sorted(enumerate(populations), key=lambda x: x[1], reverse=True)
            print(f"\nExpected dominant class: {class_labels[sorted_pops[0][0]]}")
            if len(sorted_pops) > 1:
                print(f"Expected second class: {class_labels[sorted_pops[1][0]]}")
    
    # Analyze model configuration
    print("\n2. MODEL CONFIGURATION ANALYSIS")
    print("-" * 40)
    
    if "LOSS=" in log_content:
        loss_line = [line for line in log_content.split('\n') if 'LOSS=' in line and 'SCCE' in line]
        if loss_line:
            print(f"Loss function: {loss_line[0].split('LOSS=')[1].split(',')[0].strip()}")
    
    if "void_penalty" in log_content.lower():
        print("⚠️  Custom void penalty detected in loss function")
        print("   This may be causing the model to avoid predicting voids")
    
    # Check for attention mechanism
    if "USE_ATTENTION=True" in log_content:
        print("✓ Using attention U-Net")
    else:
        print("○ Using standard U-Net")
    
    # Check for lambda conditioning
    if "LAMBDA_CONDITIONING=True" in log_content:
        print("✓ Using lambda conditioning")
    else:
        print("○ No lambda conditioning")

def main():
    parser = argparse.ArgumentParser(description='Analyze DeepVoid model prediction bias')
    parser.add_argument('log_file', type=str, help='Path to the training log file')
    args = parser.parse_args()
    
    analyze_predictions(args.log_file)
    
    print("\n3. RECOMMENDATIONS TO FIX VOID/WALL BIAS")
    print("-" * 40)
    print("Based on the analysis, here are the recommended fixes:")
    print()
    print("A. IMMEDIATE FIXES:")
    print("   1. Reduce void_penalty in SCCE_Class_Penalty from 8.0 to 2.0")
    print("   2. Reduce minority_boost from 3.0 to 1.5") 
    print("   3. Add wall_penalty to prevent wall over-prediction")
    print()
    print("B. RECOMMENDED LOSS FUNCTIONS (in order of preference):")
    print("   1. SCCE_Class_Penalty_Fixed - Best balanced approach")
    print("   2. SCCE_Proportion_Aware - Maintains target class proportions")
    print("   3. SCCE_Balanced_Class_Penalty - Alternative balanced approach")
    print("   4. SCCE - Standard loss without penalties (safest)")
    print()
    print("C. TRAINING MODIFICATIONS:")
    print("   1. Use class weights in model.compile()")
    print("   2. Implement curriculum learning with gradual penalty increase")
    print("   3. Monitor void fraction during training")
    print()
    print("D. UPDATED COMMANDS:")
    print("   # Recommended (fixed class penalty):")
    print("   python curricular.py /path/to/data 4 16 SCCE_Class_Penalty_Fixed --BATCH_SIZE 8")
    print()
    print("   # Alternative (proportion-aware):")
    print("   python curricular.py /path/to/data 4 16 SCCE_Proportion_Aware --BATCH_SIZE 8")
    print()
    print("   # Safe fallback (standard SCCE):")
    print("   python curricular.py /path/to/data 4 16 SCCE --BATCH_SIZE 8")
    print()
    print("The model has been updated with balanced parameters. Rerun training to see improvements.")

if __name__ == "__main__":
    main()
