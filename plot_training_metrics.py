#!/usr/bin/env python3
"""
Publication-Ready Training Metrics Visualization

This script reads training logs from curricular training runs and generates 
publication-quality figures showing loss, void F1, and MCC metrics over epochs.

Supports multiple input formats:
- TensorBoard event files (preferred)
- CSV files from TensorBoard exports
- Training log files with structured output

Usage:
python plot_training_metrics.py MODEL_NAME ROOT_DIR [options]

Example:
python plot_training_metrics.py TNG_curricular_SCCE_Proportion_Aware_D4_F16_attention_g-r_2025-08-28_22-28-37 /content/drive/MyDrive/ --save_format png pdf --dpi 300

Features:
- Automatic detection of training log format
- Publication-ready matplotlib styling
- Curricular training stage visualization
- Customizable figure layout and styling
- Multiple output formats (PNG, PDF, SVG)
- Comprehensive metric tracking (loss, accuracy, F1, MCC, void F1)
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
required = parser.add_argument_group('required arguments')
required.add_argument('MODEL_NAME', type=str, help='Name of the trained model.')
required.add_argument('ROOT_DIR', type=str, help='Root directory containing logs and figures.')

optional = parser.add_argument_group('optional arguments')
optional.add_argument('--log_source', type=str, default='auto',
                      choices=['auto', 'tensorboard', 'csv', 'logfile'],
                      help='Source of training logs. Default: auto (detect automatically)')
optional.add_argument('--save_format', nargs='+', default=['png'],
                      choices=['png', 'pdf', 'svg', 'eps'],
                      help='Output format(s) for figures. Default: png')
optional.add_argument('--dpi', type=int, default=300,
                      help='DPI for raster output formats. Default: 300')
optional.add_argument('--figsize', nargs=2, type=float, default=[15, 10],
                      help='Figure size in inches (width height). Default: 15 10')
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
optional.add_argument('--include_validation', action='store_true',
                      help='Include validation metrics (if available)')
optional.add_argument('--title_prefix', type=str, default='',
                      help='Prefix for figure titles. Default: empty')

args = parser.parse_args()

MODEL_NAME = args.MODEL_NAME
ROOT_DIR = args.ROOT_DIR
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

print(f'Model: {MODEL_NAME}')
print(f'Root directory: {ROOT_DIR}')
print(f'Log source: {LOG_SOURCE}')
print(f'Output formats: {SAVE_FORMAT}')
print(f'Figure size: {FIGSIZE}')
print(f'Metrics to plot: {METRICS}')

#================================================================
# Set up paths
#================================================================
LOGS_PATH = os.path.join(ROOT_DIR, 'logs', 'fit')
FIG_PATH = os.path.join(ROOT_DIR, 'figs', 'training_metrics')
os.makedirs(FIG_PATH, exist_ok=True)

#================================================================
# Publication-ready styling
#================================================================
def setup_publication_style():
    """Set up publication-ready matplotlib styling."""
    if STYLE == 'publication':
        # Custom publication style
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',
            'figure.edgecolor': 'none',
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
            'mathtext.fontset': 'dejavuserif',
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'lines.linewidth': 2.0,
            'lines.markersize': 6,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'xtick.minor.width': 0.8,
            'ytick.minor.width': 0.8,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.axisbelow': True,
            'legend.frameon': True,
            'legend.fancybox': True,
            'legend.shadow': False,
            'legend.framealpha': 0.9,
            'savefig.dpi': DPI,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    else:
        plt.style.use(STYLE)
    
    print(f'Applied {STYLE} style for publication-ready plots')

#================================================================
# Data loading functions
#================================================================
def find_log_files(model_name, logs_path):
    """Find all possible log file locations for a model."""
    log_files = {
        'tensorboard': [],
        'csv': [],
        'logfile': []
    }
    
    # Search for TensorBoard event files
    if TENSORBOARD_AVAILABLE:
        tb_pattern = os.path.join(logs_path, f'{model_name}*')
        import glob
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
    import glob
    for pattern in csv_patterns:
        log_files['csv'].extend(glob.glob(pattern))
    
    # Search for log files
    log_patterns = [
        os.path.join(ROOT_DIR, f'{model_name}*.log'),
        os.path.join(ROOT_DIR, '_stage', f'*{model_name[-20:]}*.log'),  # Match end of model name
        os.path.join(ROOT_DIR, 'logs', f'{model_name}*.log')
    ]
    for pattern in log_patterns:
        log_files['logfile'].extend(glob.glob(pattern))
    
    return log_files

def parse_tensorboard_logs(event_files):
    """Parse TensorBoard event files to extract training metrics."""
    print(f'Parsing {len(event_files)} TensorBoard event file(s)...')
    
    all_metrics = {}
    
    for event_file in event_files:
        try:
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            
            # Get available scalar tags
            scalar_tags = ea.Tags()['scalars']
            print(f'Available metrics: {scalar_tags}')
            
            for tag in scalar_tags:
                scalar_events = ea.Scalars(tag)
                epochs = [event.step for event in scalar_events]
                values = [event.value for event in scalar_events]
                
                if tag not in all_metrics:
                    all_metrics[tag] = {'epoch': [], 'value': []}
                
                all_metrics[tag]['epoch'].extend(epochs)
                all_metrics[tag]['value'].extend(values)
        
        except Exception as e:
            print(f'Error parsing {event_file}: {e}')
    
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
    
    metrics_data = []
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            print(f'Parsing log file: {log_file}')
            
            # Parse curricular training stages and epochs
            # Look for patterns like: "32/32 - 204s - 6s/step - accuracy: 0.5800 - f1_micro: 0.8548 - loss: 1.2559..."
            epoch_pattern = r'(\d+)/\d+ - \d+s.*?accuracy:\s*([\d.]+).*?f1_micro:\s*([\d.]+).*?loss:\s*([\d.]+).*?mcc:\s*([\d.]+).*?void_f1:\s*([\d.]+)'
            val_pattern = r'val_accuracy:\s*([\d.]+).*?val_f1_micro:\s*([\d.]+).*?val_loss:\s*([\d.]+).*?val_mcc:\s*([\d.]+).*?val_void_f1:\s*([\d.]+)'
            stage_pattern = r'Starting training for interparticle separation L=([\d.]+) Mpc/h \(stage (\d+)/\d+\)'
            
            current_stage = 1
            current_stage_lambda = "0.33"
            global_epoch = 0
            
            lines = content.split('\n')
            
            for line in lines:
                # Check for stage transitions
                stage_match = re.search(stage_pattern, line)
                if stage_match:
                    current_stage_lambda = stage_match.group(1)
                    current_stage = int(stage_match.group(2))
                    print(f'Found stage {current_stage}: L={current_stage_lambda} Mpc/h')
                    continue
                
                # Parse epoch metrics
                epoch_match = re.search(epoch_pattern, line)
                val_match = re.search(val_pattern, line)
                
                if epoch_match:
                    global_epoch += 1
                    
                    metrics = {
                        'epoch': global_epoch,
                        'stage': current_stage,
                        'lambda': current_stage_lambda,
                        'accuracy': float(epoch_match.group(2)),
                        'f1_micro': float(epoch_match.group(3)),
                        'loss': float(epoch_match.group(4)),
                        'mcc': float(epoch_match.group(5)),
                        'void_f1': float(epoch_match.group(6))
                    }
                    
                    # Add validation metrics if available
                    if val_match:
                        metrics.update({
                            'val_accuracy': float(val_match.group(1)),
                            'val_f1_micro': float(val_match.group(2)),
                            'val_loss': float(val_match.group(3)),
                            'val_mcc': float(val_match.group(4)),
                            'val_void_f1': float(val_match.group(5))
                        })
                    
                    metrics_data.append(metrics)
            
            if metrics_data:
                df = pd.DataFrame(metrics_data)
                print(f'Parsed {len(df)} epochs from log file')
                print(f'Stages found: {sorted(df["stage"].unique())}')
                print(f'Lambda values: {sorted(df["lambda"].unique())}')
                return df
                
        except Exception as e:
            print(f'Error parsing {log_file}: {e}')
    
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
           label=f'Training {metric.replace("_", " ").title()}', 
           linewidth=2.5, alpha=0.9)
    
    # Plot validation metric if available and requested
    if include_val and f'val_{metric}' in df.columns:
        ax.plot(df['epoch'], val_data, 
               label=f'Validation {metric.replace("_", " ").title()}', 
               linewidth=2.5, alpha=0.9, linestyle='--')
    
    # Add stage boundaries for curricular training
    if SHOW_STAGES:
        stage_boundaries = get_stage_boundaries(df)
        for boundary in stage_boundaries:
            ax.axvline(x=df['epoch'].iloc[boundary], 
                      color='red', linestyle=':', alpha=0.7, linewidth=1.5)
    
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    if MAX_EPOCHS:
        ax.set_xlim(0, MAX_EPOCHS)

def create_combined_figure(df):
    """Create a combined figure with multiple metric subplots."""
    
    metrics_config = {
        'loss': {'title': 'Training Loss', 'ylabel': 'Loss'},
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
    
    # Determine subplot layout
    if n_metrics <= 2:
        nrows, ncols = 1, n_metrics
        figsize = (FIGSIZE[0], FIGSIZE[1] // 2)
    elif n_metrics <= 4:
        nrows, ncols = 2, 2
        figsize = FIGSIZE
    else:
        nrows, ncols = 3, 2
        figsize = (FIGSIZE[0], FIGSIZE[1] * 1.2)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(plot_metrics):
        config = metrics_config[metric]
        plot_metric_subplot(axes[i], df, metric, config['title'], config['ylabel'], 
                          INCLUDE_VALIDATION, SMOOTH_WINDOW)
    
    # Hide unused subplots
    for i in range(len(plot_metrics), len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    model_display = MODEL_NAME.replace('_', ' ')
    if TITLE_PREFIX:
        title = f'{TITLE_PREFIX}: {model_display}'
    else:
        title = f'Training Metrics: {model_display}'
    
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    # Add stage information if available
    if SHOW_STAGES and 'stage' in df.columns:
        n_stages = df['stage'].nunique()
        stage_info = f'Curricular Training: {n_stages} stages'
        if 'lambda' in df.columns:
            lambdas = sorted(df['lambda'].unique())
            stage_info += f' (λ: {", ".join(lambdas)} Mpc/h)'
        
        fig.text(0.5, 0.02, stage_info, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    return fig

def create_separate_figures(df):
    """Create separate figures for each metric."""
    
    metrics_config = {
        'loss': {'title': 'Training Loss', 'ylabel': 'Loss'},
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
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
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
    log_files = find_log_files(MODEL_NAME, LOGS_PATH)
    
    print(f'Found log files:')
    for log_type, files in log_files.items():
        print(f'  {log_type}: {len(files)} files')
        for file in files:
            print(f'    - {file}')
    
    # Determine log source and load data
    df = None
    
    if LOG_SOURCE == 'auto':
        # Try in order of preference: TensorBoard -> CSV -> Log file
        if log_files['tensorboard'] and TENSORBOARD_AVAILABLE:
            df = parse_tensorboard_logs(log_files['tensorboard'])
        
        if df is None and log_files['csv']:
            df = parse_csv_logs(log_files['csv'])
        
        if df is None and log_files['logfile']:
            df = parse_log_file(log_files['logfile'])
    
    elif LOG_SOURCE == 'tensorboard':
        if not TENSORBOARD_AVAILABLE:
            raise ValueError("TensorBoard not available. Install with: pip install tensorboard")
        df = parse_tensorboard_logs(log_files['tensorboard'])
    
    elif LOG_SOURCE == 'csv':
        df = parse_csv_logs(log_files['csv'])
    
    elif LOG_SOURCE == 'logfile':
        df = parse_log_file(log_files['logfile'])
    
    if df is None or df.empty:
        raise ValueError(f"No training data could be loaded for model: {MODEL_NAME}")
    
    print(f'Successfully loaded training data: {len(df)} epochs')
    print(f'Available metrics: {list(df.columns)}')
    
    # Limit epochs if requested
    if MAX_EPOCHS:
        df = df[df['epoch'] <= MAX_EPOCHS]
        print(f'Limited to first {MAX_EPOCHS} epochs: {len(df)} data points')
    
    # Create figures
    print('Creating publication-ready figures...')
    
    if SEPARATE_FIGURES:
        figures = create_separate_figures(df)
        base_name = f'{MODEL_NAME}_metrics'
    else:
        figures = create_combined_figure(df)
        base_name = f'{MODEL_NAME}_training_metrics'
    
    # Save figures
    print('Saving figures...')
    save_figures(figures, base_name)
    
    print(f'All figures saved to: {FIG_PATH}')
    print('Training metrics visualization complete!')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(1)
