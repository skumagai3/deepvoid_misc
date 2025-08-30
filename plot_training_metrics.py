#!/usr/bin/env python3
"""
Publication-Ready Training Metrics Visualization

This script reads training logs from curricular training runs and generates 
publication-quality figures showing loss, void F1, and MCC metrics over epochs.

Supports multiple input formats:
- TensorBoard event files (optional, often empty/useless)
- CSV files from TensorBoard exports
- Training log files with structured output (preferred)
- Direct log file input

Supports multiple validation strategies:
- gradual: Uses standard val_* metrics (most common)
- hybrid: Uses val_target_* metrics as main validation (has both target and stage)

Usage:
python plot_training_metrics.py MODEL_NAME ROOT_DIR [options]
python plot_training_metrics.py --log_file /path/to/log_file.log [options]

Example:
python plot_training_metrics.py TNG_curricular_SCCE_Proportion_Aware_D4_F16_attention_g-r_2025-08-28_22-28-37 /content/drive/MyDrive/ --save_format png pdf --dpi 300
python plot_training_metrics.py --log_file /content/drive/MyDrive/logs/stdout/2025-08-19_02:09_curr_out.log --save_format png

Features:
- Automatic detection of training log format and validation strategy
- Publication-ready matplotlib styling
- Curricular training stage visualization
- Customizable figure layout and styling
- Multiple output formats (PNG, PDF, SVG)
- Comprehensive metric tracking (loss, accuracy, F1, MCC, void F1)
- Direct log file input support
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from pathlib import Path
import re
from datetime import datetime
import warnings
import glob
import traceback
warnings.filterwarnings('ignore')

# Try importing TensorBoard for event file reading
try:
    import tensorflow as tf
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorBoard not available. Will only support CSV and log file formats.")
    TENSORBOARD_AVAILABLE = False

print('>>> Running plot_training_metrics.py')

#================================================================
# Parse command line arguments
#================================================================
parser = argparse.ArgumentParser(description='Generate publication-ready training metrics plots.')

# Create mutually exclusive group for input methods
input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument('--log_file', type=str, 
                        help='Direct path to log file to parse')
input_group.add_argument('MODEL_NAME', nargs='?', type=str, 
                        help='Name of the trained model (requires ROOT_DIR)')

parser.add_argument('ROOT_DIR', nargs='?', type=str, 
                   help='Root directory containing logs and figures (required with MODEL_NAME)')

optional = parser.add_argument_group('optional arguments')
optional.add_argument('--log_source', type=str, default='auto',
                      choices=['auto', 'tensorboard', 'csv', 'logfile'],
                      help='Source of training logs. Default: auto (detect automatically)')
optional.add_argument('--save_format', nargs='+', default=['png'],
                      choices=['png', 'pdf', 'svg', 'eps'],
                      help='Output format(s) for figures. Default: png')
optional.add_argument('--dpi', type=int, default=300,
                      help='DPI for raster output formats. Default: 300')
optional.add_argument('--figsize', nargs=2, type=float, default=[12, 4],
                      help='Figure size in inches (width height). Default: 12 4')
optional.add_argument('--style', type=str, default='publication',
                      choices=['publication', 'seaborn', 'ggplot', 'classic'],
                      help='Plot style. Default: publication')
optional.add_argument('--show_stages', action='store_true',
                      help='Highlight curricular training stages with vertical lines')
optional.add_argument('--smooth_window', type=int, default=0,
                      help='Window size for smoothing metrics (0 = no smoothing). Default: 0')
optional.add_argument('--max_epochs', type=int, default=None,
                      help='Maximum number of epochs to plot. Default: all')
optional.add_argument('--metrics', nargs='+', default=['loss', 'void_f1', 'mcc'],
                      choices=['loss', 'accuracy', 'f1_micro', 'mcc', 'void_f1', 'all'],
                      help='Metrics to plot. Default: loss void_f1 mcc')
optional.add_argument('--separate_figures', action='store_true',
                      help='Create separate figure for each metric instead of subplots')
optional.add_argument('--include_validation', action='store_true', default=True,
                      help='Include validation metrics (if available)')
optional.add_argument('--title_prefix', type=str, default='',
                      help='Prefix for figure titles. Default: empty')
optional.add_argument('--output_dir', type=str, default=None,
                      help='Output directory for figures (default: same as log file or ROOT_DIR/figs/training_metrics)')

args = parser.parse_args()

# Validate arguments
if args.MODEL_NAME and not args.ROOT_DIR:
    parser.error("ROOT_DIR is required when using MODEL_NAME")
if args.log_file and args.ROOT_DIR:
    parser.error("Cannot specify both --log_file and ROOT_DIR")

# Set up variables based on input method
if args.log_file:
    # Direct log file mode
    LOG_FILE_PATH = args.log_file
    MODEL_NAME = os.path.splitext(os.path.basename(LOG_FILE_PATH))[0]
    ROOT_DIR = os.path.dirname(os.path.dirname(LOG_FILE_PATH))  # Go up two levels from logs/stdout
    print(f"Direct log file mode: {LOG_FILE_PATH}")
    print(f"Inferred model name: {MODEL_NAME}")
    print(f"Inferred root directory: {ROOT_DIR}")
else:
    # Model name mode
    MODEL_NAME = args.MODEL_NAME
    ROOT_DIR = args.ROOT_DIR
    LOG_FILE_PATH = None
    print(f"Model name mode: {MODEL_NAME}")

LOG_SOURCE = args.log_source
SAVE_FORMAT = args.save_format
DPI = args.dpi
FIGSIZE = tuple(args.figsize)
STYLE = args.style
SHOW_STAGES = args.show_stages
SMOOTH_WINDOW = args.smooth_window
MAX_EPOCHS = args.max_epochs
METRICS = args.metrics
SEPARATE_FIGURES = args.separate_figures
INCLUDE_VALIDATION = args.include_validation
TITLE_PREFIX = args.title_prefix

print(f'Root directory: {ROOT_DIR}')
print(f'Log source: {LOG_SOURCE}')
print(f'Output formats: {SAVE_FORMAT}')
print(f'Figure size: {FIGSIZE}')
print(f'Metrics to plot: {METRICS}')

#================================================================
# Set up paths
#================================================================
LOGS_PATH = os.path.join(ROOT_DIR, 'logs', 'fit')

# Set up output directory
if args.output_dir:
    FIG_PATH = args.output_dir
elif LOG_FILE_PATH:
    # For direct log file mode, save figures next to the log file
    FIG_PATH = os.path.join(os.path.dirname(LOG_FILE_PATH), 'figures')
else:
    # For model name mode, use standard path
    FIG_PATH = os.path.join(ROOT_DIR, 'figs', 'training_metrics')

os.makedirs(FIG_PATH, exist_ok=True)
print(f'Figures will be saved to: {FIG_PATH}')

#================================================================
# Publication-ready styling
#================================================================
def setup_publication_style():
    """Set up publication-ready matplotlib styling."""
    if STYLE == 'publication':
        # Set global font sizes
        plt.rc('font', size=12)          # Default text size
        plt.rc('axes', titlesize=16)     # Title font size
        plt.rc('axes', labelsize=14)     # Axis label font size
        plt.rc('xtick', labelsize=12)    # X-tick label font size
        plt.rc('ytick', labelsize=12)    # Y-tick label font size
        plt.rc('legend', fontsize=12)    # Legend font size
    else:
        plt.style.use(STYLE)
    
    print(f'Applied {STYLE} style for publication-ready plots')

#================================================================
# Data loading functions
#================================================================
def find_log_files(model_name, logs_path, direct_log_file=None):
    """Find all possible log file locations for a model or use direct log file."""
    log_files = {
        'tensorboard': [],
        'csv': [],
        'logfile': []
    }
    
    # If direct log file is provided, use it directly
    if direct_log_file:
        if os.path.exists(direct_log_file):
            log_files['logfile'].append(direct_log_file)
            print(f'Using direct log file: {direct_log_file}')
            return log_files
        else:
            print(f'Direct log file not found: {direct_log_file}')
            return log_files
    
    # Search for TensorBoard event files
    if TENSORBOARD_AVAILABLE:
        tb_pattern = os.path.join(logs_path, f'{model_name}*')
        for tb_dir in glob.glob(tb_pattern):
            if os.path.isdir(tb_dir):
                for root, dirs, files in os.walk(tb_dir):
                    for file in files:
                        if file.startswith('events.out.tfevents'):
                            log_files['tensorboard'].append(os.path.join(root, file))
    
    # Search for CSV files
    csv_patterns = [
        os.path.join(ROOT_DIR, f'{model_name}*training*.csv'),
        os.path.join(ROOT_DIR, f'{model_name}*metrics*.csv'),
        os.path.join(ROOT_DIR, 'training_logs', f'{model_name}*.csv')
    ]
    for pattern in csv_patterns:
        log_files['csv'].extend(glob.glob(pattern))
    
    # Extract datetime from model name and find exact matching log file
    # Model name format: TNG_curricular_SCCE_Proportion_Aware_D4_F16_RSD_RSDrot_attention_g-r_2025-08-18_17-16-07
    datetime_match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})$', model_name)
    
    if datetime_match:
        date_part = datetime_match.group(1)  # 2025-08-18
        hour = int(datetime_match.group(2))  # 17
        minute = int(datetime_match.group(3))  # 16
        
        # Convert to log file format: 2025-08-18_17:16_curr_out.log
        log_datetime = f'{date_part}_{hour:02d}:{minute:02d}'
        target_log = os.path.join(ROOT_DIR, 'logs', 'stdout', f'{log_datetime}_curr_out.log')
        
        print(f'Looking for specific log file: {target_log}')
        
        if os.path.exists(target_log):
            log_files['logfile'].append(target_log)
            print(f'Found exact match: {target_log}')
        else:
            print(f'Exact match not found: {target_log}')
            print(f'Searching for log files within ±2 minutes...')
            
            # Check what log files actually exist for this date
            log_dir = os.path.join(ROOT_DIR, 'logs', 'stdout')
            if os.path.exists(log_dir):
                all_logs = glob.glob(os.path.join(log_dir, f'{date_part}_*_curr_out.log'))
                print(f'Available log files for {date_part}:')
                for log in sorted(all_logs):
                    print(f'  - {os.path.basename(log)}')
                
                # Try to find log files within ±2 minutes tolerance
                found_within_tolerance = False
                for time_offset in range(-2, 3):  # -2, -1, 0, 1, 2 minutes
                    target_minute = minute + time_offset
                    target_hour = hour
                    
                    # Handle minute overflow/underflow
                    if target_minute >= 60:
                        target_minute -= 60
                        target_hour += 1
                    elif target_minute < 0:
                        target_minute += 60
                        target_hour -= 1
                    
                    # Handle hour overflow/underflow (basic check)
                    if target_hour >= 24:
                        target_hour = 0
                    elif target_hour < 0:
                        target_hour = 23
                    
                    # Try different time formats
                    time_formats = [
                        f'{target_hour:02d}:{target_minute:02d}',  # HH:MM
                        f'{target_hour:02d}-{target_minute:02d}',  # HH-MM
                        f'{target_hour:02d}_{target_minute:02d}',  # HH_MM
                    ]
                    
                    for time_format in time_formats:
                        candidate_log = os.path.join(log_dir, f'{date_part}_{time_format}_curr_out.log')
                        if os.path.exists(candidate_log):
                            log_files['logfile'].append(candidate_log)
                            offset_str = f"+{time_offset}" if time_offset > 0 else str(time_offset)
                            print(f'Found log file within tolerance ({offset_str} min): {os.path.basename(candidate_log)}')
                            found_within_tolerance = True
                            break
                    
                    if found_within_tolerance:
                        break
                
                if not found_within_tolerance:
                    print(f'No log files found within ±2 minutes of {hour:02d}:{minute:02d}')
            else:
                print(f'Log directory does not exist: {log_dir}')
    else:
        print(f'Could not extract datetime from model name: {model_name}')
        # Fallback to broader search only if datetime extraction failed
        log_patterns = [
            os.path.join(ROOT_DIR, 'logs', 'stdout', f'{model_name}*.log'),
            os.path.join(ROOT_DIR, 'logs', 'stdout', f'*{model_name}*.log'),
        ]
        for pattern in log_patterns:
            matches = glob.glob(pattern)
            log_files['logfile'].extend(matches)
    
    # Remove duplicates
    log_files['logfile'] = list(set(log_files['logfile']))
    
    return log_files
    # Model name format: TNG_curricular_SCCE_Proportion_Aware_D4_F16_RSD_RSDrot_attention_g-r_2025-08-18_17-16-07
    import re
    datetime_match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})$', model_name)
    
    if datetime_match:
        date_part = datetime_match.group(1)  # 2025-08-18
        hour = int(datetime_match.group(2))  # 17
        minute = int(datetime_match.group(3))  # 16
        
        # Convert to log file format: 2025-08-18_17:16_curr_out.log
        log_datetime = f'{date_part}_{hour:02d}:{minute:02d}'
        target_log = os.path.join(ROOT_DIR, 'logs', 'stdout', f'{log_datetime}_curr_out.log')
        
        print(f'Looking for specific log file: {target_log}')
        
        if os.path.exists(target_log):
            log_files['logfile'].append(target_log)
            print(f'Found exact match: {target_log}')
        else:
            print(f'Exact match not found: {target_log}')
            print(f'Searching for log files within ±2 minutes...')
            
            # Check what log files actually exist for this date
            log_dir = os.path.join(ROOT_DIR, 'logs', 'stdout')
            if os.path.exists(log_dir):
                import glob
                all_logs = glob.glob(os.path.join(log_dir, f'{date_part}_*_curr_out.log'))
                print(f'Available log files for {date_part}:')
                for log in sorted(all_logs):
                    print(f'  - {os.path.basename(log)}')
                
                # Try to find log files within ±2 minutes tolerance
                found_within_tolerance = False
                for time_offset in range(-2, 3):  # -2, -1, 0, 1, 2 minutes
                    target_minute = minute + time_offset
                    target_hour = hour
                    
                    # Handle minute overflow/underflow
                    if target_minute >= 60:
                        target_minute -= 60
                        target_hour += 1
                    elif target_minute < 0:
                        target_minute += 60
                        target_hour -= 1
                    
                    # Handle hour overflow/underflow (basic check)
                    if target_hour >= 24:
                        target_hour = 0
                    elif target_hour < 0:
                        target_hour = 23
                    
                    # Try different time formats
                    time_formats = [
                        f'{target_hour:02d}:{target_minute:02d}',  # HH:MM
                        f'{target_hour:02d}-{target_minute:02d}',  # HH-MM
                        f'{target_hour:02d}_{target_minute:02d}',  # HH_MM
                    ]
                    
                    for time_format in time_formats:
                        candidate_log = os.path.join(log_dir, f'{date_part}_{time_format}_curr_out.log')
                        if os.path.exists(candidate_log):
                            log_files['logfile'].append(candidate_log)
                            offset_str = f"+{time_offset}" if time_offset > 0 else str(time_offset)
                            print(f'Found log file within tolerance ({offset_str} min): {os.path.basename(candidate_log)}')
                            found_within_tolerance = True
                            break
                    
                    if found_within_tolerance:
                        break
                
                if not found_within_tolerance:
                    print(f'No log files found within ±2 minutes of {hour:02d}:{minute:02d}')
            else:
                print(f'Log directory does not exist: {log_dir}')
    else:
        print(f'Could not extract datetime from model name: {model_name}')
        # Fallback to broader search only if datetime extraction failed
        log_patterns = [
            os.path.join(ROOT_DIR, 'logs', 'stdout', f'{model_name}*.log'),
            os.path.join(ROOT_DIR, 'logs', 'stdout', f'*{model_name}*.log'),
        ]
        for pattern in log_patterns:
            matches = glob.glob(pattern)
            log_files['logfile'].extend(matches)
    
    # Remove duplicates
    log_files['logfile'] = list(set(log_files['logfile']))
    
    return log_files

def parse_tensorboard_logs(event_files):
    """Parse TensorBoard event files to extract training metrics."""
    print(f'Parsing {len(event_files)} TensorBoard event file(s)...')
    
    all_metrics = {}
    files_with_data = 0
    
    for event_file in event_files:
        try:
            ea = event_accumulator.EventAccumulator(
                event_file,
                size_guidance={
                    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                    event_accumulator.IMAGES: 4,
                    event_accumulator.AUDIO: 4,
                    event_accumulator.SCALARS: 0,  # 0 means load all
                    event_accumulator.HISTOGRAMS: 1,
                }
            )
            ea.Reload()
            
            # Get available scalar tags
            tags = ea.Tags()
            if 'scalars' in tags and tags['scalars']:
                scalar_tags = tags['scalars']
                print(f'File {os.path.basename(event_file)}: {len(scalar_tags)} metrics found: {scalar_tags}')
                files_with_data += 1
                
                for tag in scalar_tags:
                    scalar_events = ea.Scalars(tag)
                    epochs = [event.step for event in scalar_events]
                    values = [event.value for event in scalar_events]
                    
                    if tag not in all_metrics:
                        all_metrics[tag] = {'epoch': [], 'value': []}
                    
                    all_metrics[tag]['epoch'].extend(epochs)
                    all_metrics[tag]['value'].extend(values)
            else:
                print(f'File {os.path.basename(event_file)}: No scalar metrics found')
        
        except Exception as e:
            print(f'Error parsing {event_file}: {e}')
    
    print(f'Successfully parsed {files_with_data}/{len(event_files)} TensorBoard files')
    
    # Convert to DataFrame
    if all_metrics:
        # Find the metric with the most data points to use as reference
        max_epochs = max(len(data['epoch']) for data in all_metrics.values())
        reference_metric = next(name for name, data in all_metrics.items() 
                              if len(data['epoch']) == max_epochs)
        
        df = pd.DataFrame({'epoch': all_metrics[reference_metric]['epoch']})
        
        for metric_name, data in all_metrics.items():
            # Interpolate to match reference epochs if needed
            if len(data['epoch']) == len(df):
                df[metric_name] = data['value']
            else:
                # Use interpolation for mismatched lengths
                df[metric_name] = np.interp(df['epoch'], data['epoch'], data['value'])
        
        print(f'Loaded TensorBoard data: {len(df)} epochs, {len(df.columns)-1} metrics')
        return df
    else:
        print('No metrics data found in TensorBoard files')
    
    return None

def parse_csv_logs(csv_files):
    """Parse CSV files containing training metrics."""
    print(f'Parsing {len(csv_files)} CSV file(s)...')
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            print(f'Loaded CSV data from {csv_file}: {len(df)} epochs, {len(df.columns)} columns')
            print(f'Available columns: {list(df.columns)}')
            return df
        except Exception as e:
            print(f'Error parsing {csv_file}: {e}')
    
    return None

def parse_log_file(log_files):
    """Parse structured log files to extract training metrics."""
    print(f'Parsing {len(log_files)} log file(s)...')
    
    all_metrics_data = []
    
    for log_file in log_files:
        try:
            print(f'Checking log file: {log_file}')
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            print(f'Parsing log file: {log_file} ({len(content)} characters)')
            
            # Parse epoch metrics - handle different validation strategies based on actual log formats
            
            # Pattern 1: Gradual validation strategy (standard val_ metrics)
            # Example: "32/32 - 85s - 3s/step - accuracy: 0.2058 - f1_micro: 0.5748 - loss: 1.8958 - mcc: 0.0693 - val_accuracy: 0.0901 - val_f1_micro: 0.7293 - val_loss: 5.2175 - val_mcc: -6.0898e-03 - val_void_f1: 4.5386e-04 - void_f1: 0.0068 - learning_rate: 6.0000e-05"
            gradual_pattern = r'(\d+)/\d+.*?- accuracy:\s*([\d.]+).*?- f1_micro:\s*([\d.]+).*?- loss:\s*([\d.]+).*?- mcc:\s*([\d.-]+e?-?\d*).*?- val_accuracy:\s*([\d.]+).*?- val_f1_micro:\s*([\d.]+).*?- val_loss:\s*([\d.]+).*?- val_mcc:\s*([\d.-]+e?-?\d*).*?- val_void_f1:\s*([\d.e-]+).*?- void_f1:\s*([\d.]+)'
            
            # Pattern 2: Hybrid validation strategy (has both target and stage metrics)
            # Example: "13/13 - 308s - 24s/step - accuracy: 0.5031 - f1_micro: 0.8952 - loss: 1.4201 - mcc: 0.2255 - void_f1: 0.6919 - learning_rate: 6.0000e-05 - val_hybrid_loss: 42.3378 - val_target_loss: 41.8492 - val_stage_loss: 42.5472 - val_hybrid_alpha: 0.3000 - val_target_accuracy: 0.0939 - val_stage_accuracy: 0.0812 - val_target_mcc: -7.6302e-03 - val_stage_mcc: 1.8706e-04 - val_target_f1_micro: 0.7365 - val_stage_f1_micro: 0.7314 - val_target_void_f1: 2.2051e-04 - val_stage_void_f1: 1.2425e-04 - val_loss_divergence: 0.0164"
            hybrid_pattern = r'(\d+)/\d+.*?- accuracy:\s*([\d.]+).*?- f1_micro:\s*([\d.]+).*?- loss:\s*([\d.]+).*?- mcc:\s*([\d.-]+e?-?\d*).*?- void_f1:\s*([\d.]+).*?val_target_accuracy:\s*([\d.]+).*?val_target_f1_micro:\s*([\d.]+).*?val_target_loss:\s*([\d.]+).*?val_target_mcc:\s*([\d.-]+e?-?\d*).*?val_target_void_f1:\s*([\d.e-]+)'
            
            # Note: Based on the log files examined, there appear to be only two validation strategies:
            # 1. Gradual: standard val_* metrics
            # 2. Hybrid: both val_target_* and val_stage_* metrics (always together)
            # Pure "target" or "stage" validation strategies were not found in the actual logs.
            
            stage_pattern = r'Starting training for interparticle separation L=([\d.]+) Mpc/h \(stage (\d+)/\d+\)'
            
            current_stage = 1
            current_stage_lambda = "0.33"
            global_epoch = 0
            found_any_metrics = False
            
            lines = content.split('\n')
            current_train_metrics = None
            
            for line_num, line in enumerate(lines):
                # Check for stage transitions
                stage_match = re.search(stage_pattern, line)
                if stage_match:
                    current_stage_lambda = stage_match.group(1)
                    current_stage = int(stage_match.group(2))
                    print(f'Found stage {current_stage}: L={current_stage_lambda} Mpc/h')
                    continue
                
                # Try different patterns to handle the two main validation strategies
                epoch_match = None
                validation_strategy = None
                
                # Try gradual strategy first (most common - standard val_ metrics)
                epoch_match = re.search(gradual_pattern, line)
                if epoch_match:
                    validation_strategy = 'gradual'
                else:
                    # Try hybrid strategy (target + stage validation)
                    epoch_match = re.search(hybrid_pattern, line)
                    if epoch_match:
                        validation_strategy = 'hybrid'
                
                if epoch_match:
                    global_epoch += 1
                    found_any_metrics = True
                    
                    # Base training metrics (common to all patterns)
                    metrics_data = {
                        'epoch': global_epoch,
                        'stage': current_stage,
                        'lambda': current_stage_lambda,
                        'accuracy': float(epoch_match.group(2)),
                        'f1_micro': float(epoch_match.group(3)),
                        'loss': float(epoch_match.group(4)),
                        'mcc': float(epoch_match.group(5)),
                        'void_f1': float(epoch_match.group(6))
                    }
                    
                    # Add validation metrics based on strategy
                    if validation_strategy == 'gradual':
                        # Standard val_ metrics
                        metrics_data.update({
                            'val_accuracy': float(epoch_match.group(6)),
                            'val_f1_micro': float(epoch_match.group(7)),
                            'val_loss': float(epoch_match.group(8)),
                            'val_mcc': float(epoch_match.group(9)),
                            'val_void_f1': float(epoch_match.group(10))
                        })
                        # void_f1 is group 11 for gradual pattern
                        metrics_data['void_f1'] = float(epoch_match.group(11))
                    elif validation_strategy == 'hybrid':
                        # Use target validation metrics as the main validation metrics for hybrid
                        # (this gives us the most relevant validation performance)
                        metrics_data.update({
                            'val_accuracy': float(epoch_match.group(7)),
                            'val_f1_micro': float(epoch_match.group(8)),
                            'val_loss': float(epoch_match.group(9)),
                            'val_mcc': float(epoch_match.group(10)),
                            'val_void_f1': float(epoch_match.group(11))
                        })
                    
                    all_metrics_data.append(metrics_data)
                    
                    if global_epoch <= 5:  # Debug first few epochs
                        val_info = f"val_loss={metrics_data.get('val_loss', 'N/A')}" if 'val_loss' in metrics_data else "no validation"
                        print(f'Epoch {global_epoch}: loss={metrics_data["loss"]:.4f}, void_f1={metrics_data["void_f1"]:.4f}, mcc={metrics_data["mcc"]:.4f} ({validation_strategy} strategy, {val_info})')
                    
                    if validation_strategy and global_epoch == 1:
                        print(f'Detected validation strategy: {validation_strategy}')
                        print(f'Validation metrics available: {[k for k in metrics_data.keys() if k.startswith("val_")]}')
            
            if found_any_metrics:
                print(f'Successfully parsed {global_epoch} epochs from {log_file}')
                
                # Debug: Show sample lines to understand validation format
                print('Debug: Looking for validation metrics format...')
                val_lines = [line for line in lines if 'val_' in line]
                if val_lines:
                    print(f'Found {len(val_lines)} lines with "val_" - showing first 3:')
                    for i, line in enumerate(val_lines[:3]):
                        print(f'  Val line {i+1}: {line.strip()}')
                else:
                    print('No lines found containing "val_" - validation metrics may not be logged')
                
                # Also check for lines that might contain validation metrics in different format
                potential_val_lines = [line for line in lines if 'validation' in line.lower() or 'val ' in line]
                if potential_val_lines:
                    print(f'Found {len(potential_val_lines)} lines with "validation" - showing first 3:')
                    for i, line in enumerate(potential_val_lines[:3]):
                        print(f'  Potential val line {i+1}: {line.strip()}')
            else:
                print(f'No training metrics found in {log_file}')
                # Show sample lines to help debug the format
                sample_lines = [line for line in lines if 'loss:' in line or 'accuracy:' in line]
                if sample_lines:
                    print(f'Sample lines with metrics (first 3):')
                    for i, line in enumerate(sample_lines[:3]):
                        print(f'  {i+1}: {line.strip()}')
                else:
                    print('No lines found containing "loss:" or "accuracy:"')
                
        except Exception as e:
            print(f'Error parsing {log_file}: {e}')
            import traceback
            traceback.print_exc()
    
    if all_metrics_data:
        df = pd.DataFrame(all_metrics_data)
        # Remove duplicates and sort by epoch
        df = df.drop_duplicates(subset=['epoch']).sort_values('epoch').reset_index(drop=True)
        
        print(f'Final result: Parsed {len(df)} unique epochs from log files')
        print(f'Stages found: {sorted(df["stage"].unique())}')
        print(f'Lambda values: {sorted(df["lambda"].unique())}')
        print(f'Metric ranges:')
        for metric in ['loss', 'void_f1', 'mcc']:
            if metric in df.columns:
                print(f'  {metric}: {df[metric].min():.4f} - {df[metric].max():.4f}')
        return df
    else:
        print('No metrics data found in any log files')
    
    return None

def smooth_metrics(data, window_size):
    """Apply moving average smoothing to metrics."""
    if window_size <= 1:
        return data
    
    return data.rolling(window=window_size, center=True, min_periods=1).mean()

#================================================================
# Plotting functions
#================================================================
def get_stage_boundaries(df):
    """Extract stage boundaries for curricular training visualization."""
    if 'stage' not in df.columns:
        return []
    
    boundaries = []
    current_stage = df['stage'].iloc[0]
    
    for i, stage in enumerate(df['stage']):
        if stage != current_stage:
            boundaries.append(i)
            current_stage = stage
    
    return boundaries

def plot_metric_subplot(ax, df, metric, title, ylabel, include_val=False, smooth_window=0):
    """Plot a single metric with optional validation data."""
    
    # Apply smoothing if requested
    if smooth_window > 0:
        train_data = smooth_metrics(df[metric], smooth_window)
        if include_val and f'val_{metric}' in df.columns:
            val_data = smooth_metrics(df[f'val_{metric}'], smooth_window)
    else:
        train_data = df[metric]
        if include_val and f'val_{metric}' in df.columns:
            val_data = df[f'val_{metric}']
    
    # Plot training metric
    ax.plot(df['epoch'], train_data, 
           label='Training', 
           linewidth=2.5, alpha=0.9)
    
    # Plot validation metric if available and requested
    if include_val and f'val_{metric}' in df.columns:
        ax.plot(df['epoch'], val_data, 
               label='Validation', 
               linewidth=2.5, alpha=0.9)
    
    # Add stage boundaries for curricular training
    if SHOW_STAGES:
        stage_boundaries = get_stage_boundaries(df)
        for boundary in stage_boundaries:
            ax.axvline(x=df['epoch'].iloc[boundary], 
                      color='red', linestyle=':', alpha=0.7, linewidth=1.5)
    
    ax.set_title(title, pad=15)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    if MAX_EPOCHS:
        ax.set_xlim(0, MAX_EPOCHS)
    
    # Cap loss y-axis to 0-3 for better visualization
    if metric == 'loss':
        ax.set_ylim(0, 3)

def create_combined_figure(df):
    """Create a combined figure with multiple metric subplots."""
    
    metrics_config = {
        'loss': {'title': 'Loss', 'ylabel': 'Loss'},
        'accuracy': {'title': 'Accuracy', 'ylabel': 'Accuracy'},
        'f1_micro': {'title': 'Micro F1 Score', 'ylabel': 'F1 Score'},
        'mcc': {'title': 'Matthews Correlation Coefficient', 'ylabel': 'MCC'},
        'void_f1': {'title': 'Void F1 Score', 'ylabel': 'Void F1'}
    }
    
    # Filter to requested metrics
    if 'all' in METRICS:
        plot_metrics = list(metrics_config.keys())
    else:
        plot_metrics = [m for m in METRICS if m in metrics_config and m in df.columns]
    
    if not plot_metrics:
        raise ValueError(f"No valid metrics found. Available: {list(df.columns)}")
    
    n_metrics = len(plot_metrics)
    
    # Force 1x3 layout for the three main metrics
    nrows, ncols = 1, n_metrics
    figsize = (5 * n_metrics, 4)  # Wider figure for 1x3 layout
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Plot each metric
    for i, metric in enumerate(plot_metrics):
        config = metrics_config[metric]
        plot_metric_subplot(axes[i], df, metric, config['title'], config['ylabel'], 
                          INCLUDE_VALIDATION, SMOOTH_WINDOW)
    
    # No suptitle - removed as requested
    
    plt.tight_layout()
    return fig

def create_separate_figures(df):
    """Create separate figures for each metric."""
    
    metrics_config = {
        'loss': {'title': 'Loss', 'ylabel': 'Loss'},
        'accuracy': {'title': 'Accuracy', 'ylabel': 'Accuracy'},
        'f1_micro': {'title': 'Micro F1 Score', 'ylabel': 'F1 Score'},
        'mcc': {'title': 'Matthews Correlation Coefficient', 'ylabel': 'MCC'},
        'void_f1': {'title': 'Void F1 Score', 'ylabel': 'Void F1'}
    }
    
    # Filter to requested metrics
    if 'all' in METRICS:
        plot_metrics = list(metrics_config.keys())
    else:
        plot_metrics = [m for m in METRICS if m in metrics_config and m in df.columns]
    
    figures = {}
    
    for metric in plot_metrics:
        config = metrics_config[metric]
        
        fig, ax = plt.subplots(1, 1, figsize=(FIGSIZE[0] // 2, FIGSIZE[1] // 2))
        
        plot_metric_subplot(ax, df, metric, config['title'], config['ylabel'], 
                          INCLUDE_VALIDATION, SMOOTH_WINDOW)
        
        # Add title
        model_display = MODEL_NAME.replace('_', ' ')
        if TITLE_PREFIX:
            title = f'{TITLE_PREFIX}: {config["title"]}'
        else:
            title = f'{config["title"]}: {model_display}'
        
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        figures[metric] = fig
    
    return figures

def save_figures(figures, base_name="training_metrics"):
    """Save figures in requested formats."""
    
    if isinstance(figures, dict):
        # Multiple figures (separate mode)
        for metric, fig in figures.items():
            for fmt in SAVE_FORMAT:
                filename = f'{base_name}_{metric}.{fmt}'
                filepath = os.path.join(FIG_PATH, filename)
                fig.savefig(filepath, dpi=DPI, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                print(f'Saved: {filepath}')
    else:
        # Single figure (combined mode)
        for fmt in SAVE_FORMAT:
            filename = f'{base_name}.{fmt}'
            filepath = os.path.join(FIG_PATH, filename)
            figures.savefig(filepath, dpi=DPI, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
            print(f'Saved: {filepath}')

#================================================================
# Main execution
#================================================================
def main():
    """Main execution function."""
    
    print('Setting up publication-ready styling...')
    setup_publication_style()
    
    print('Searching for training log files...')
    log_files = find_log_files(MODEL_NAME, LOGS_PATH, direct_log_file=LOG_FILE_PATH)
    
    print(f'Found log files:')
    for log_type, files in log_files.items():
        print(f'  {log_type}: {len(files)} files')
        for file in files:
            print(f'    - {file}')
    
    # Determine log source and load data
    df = None
    
    if LOG_SOURCE == 'auto':
        # Try in order of preference: Log file -> CSV -> TensorBoard
        # Skip TensorBoard by default as they're often empty/useless
        
        # First try log files (most reliable)
        if log_files['logfile']:
            print('Trying log files first...')
            df = parse_log_file(log_files['logfile'])
            if df is not None and not df.empty:
                print('Successfully loaded data from log files')
        
        # If log files failed, try CSV files
        if (df is None or df.empty) and log_files['csv']:
            print('Log files failed/empty, trying CSV files...')
            df = parse_csv_logs(log_files['csv'])
        
        # Last resort: try TensorBoard (often empty/useless but retain functionality)
        if (df is None or df.empty) and log_files['tensorboard'] and TENSORBOARD_AVAILABLE:
            print('Log and CSV files failed, trying TensorBoard as last resort...')
            df = parse_tensorboard_logs(log_files['tensorboard'])
            if df is not None and not df.empty:
                print('Successfully loaded data from TensorBoard')
            else:
                print('TensorBoard files found but no metrics extracted')
    
    elif LOG_SOURCE == 'tensorboard':
        if not TENSORBOARD_AVAILABLE:
            raise ValueError("TensorBoard not available. Install with: pip install tensorboard")
        df = parse_tensorboard_logs(log_files['tensorboard'])
    
    elif LOG_SOURCE == 'csv':
        df = parse_csv_logs(log_files['csv'])
    
    elif LOG_SOURCE == 'logfile':
        df = parse_log_file(log_files['logfile'])
    
    if df is None or df.empty:
        if LOG_FILE_PATH:
            raise ValueError(f"No training data could be loaded from log file: {LOG_FILE_PATH}")
        else:
            raise ValueError(f"No training data could be loaded for model: {MODEL_NAME}")
    
    print(f'Successfully loaded training data: {len(df)} epochs')
    print(f'Available metrics: {list(df.columns)}')
    
    # Limit epochs if requested
    if MAX_EPOCHS:
        df = df[df['epoch'] <= MAX_EPOCHS]
        print(f'Limited to first {MAX_EPOCHS} epochs: {len(df)} data points')
    
    # Create figures
    print('Creating publication-ready figures...')
    
    # Create a clean base name for the figures
    if LOG_FILE_PATH:
        # Extract a meaningful name from the log file
        log_basename = os.path.splitext(os.path.basename(LOG_FILE_PATH))[0]
        base_name = log_basename.replace('_curr_out', '').replace('_out', '')
    else:
        base_name = MODEL_NAME
    
    if SEPARATE_FIGURES:
        figures = create_separate_figures(df)
        final_base_name = f'{base_name}_metrics'
    else:
        figures = create_combined_figure(df)
        final_base_name = f'{base_name}_training_metrics'
    
    # Save figures
    print('Saving figures...')
    save_figures(figures, final_base_name)
    
    print(f'All figures saved to: {FIG_PATH}')
    print('Training metrics visualization complete!')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(1)
