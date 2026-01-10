#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def set_style():
    """Set a publication-quality style for matplotlib."""
    # Try to use a seaborn style if available, otherwise fallback to a clean style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('ggplot')
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def prepare_grouped_data(df: pd.DataFrame, sort_metric: str):
    """
    Prepare data for grouped plotting.
    Returns a DataFrame where each row is a base model, with columns for:
    - zero_std, five_std (Standard)
    - zero_enh, five_enh (Enhanced)
    """
    df = df.copy()
    df['base_name'] = df['model'].apply(lambda x: x.replace('-deberta', ''))
    df['is_enhanced'] = df['model'].apply(lambda x: '-deberta' in x)
    
    # Separate standard and enhanced
    std_df = df[~df['is_enhanced']].set_index('base_name')
    enh_df = df[df['is_enhanced']].set_index('base_name')
    
    # Merge
    merged = std_df.join(enh_df, lsuffix='_std', rsuffix='_enh', how='outer')
    
    # Sort
    # Prefer sorting by standard zero-shot metric, if available
    sort_col = f'{sort_metric}_std'
    if sort_col in merged.columns:
        merged = merged.sort_values(sort_col, ascending=True)
    
    return merged

def plot_grouped_bars(df: pd.DataFrame, metric_prefix: str, title: str, xlabel: str, output_path: Path, show_improvement=False):
    """Generic function to plot grouped bars for Standard vs Enhanced models."""
    
    # Prepare data
    # metric_prefix is like 'accuracy' or 'spearman' or 'avg_time'
    # Columns in df will be like 'zero_accuracy_std', 'five_accuracy_enh', etc.
    
    grouped_df = prepare_grouped_data(df, f'zero_{metric_prefix}')
    
    models = grouped_df.index
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Colors
    c_0_std = '#a6cee3' # Light Blue
    c_5_std = '#1f78b4' # Dark Blue
    c_0_enh = '#cab2d6' # Light Purple
    c_5_enh = '#6a3d9a' # Dark Purple
    c_deberta = '#ff7f00' # Orange
    c_smollm = '#FFB74D' # Lighter Orange

    # Helper to get data safely
    def get_data(row, col):
        val = row.get(col)
        return val if pd.notna(val) else np.nan

    # Plot bars
    # We have 4 positions: -1.5w, -0.5w, +0.5w, +1.5w
    
    for i, (model_name, row) in enumerate(grouped_df.iterrows()):
        # Determine colors for this model
        # Special cases for DeBERTa base and SmolLM
        is_deberta_base = model_name == 'deberta-finetune'
        is_smollm = 'smollm' in model_name
        
        # Standard 0-shot
        v_0_std = get_data(row, f'zero_{metric_prefix}_std')
        if pd.notna(v_0_std):
            c = c_deberta if is_deberta_base else (c_smollm if is_smollm else c_0_std)
            rect = ax.barh(i - 1.5*width, v_0_std, width, color=c, alpha=0.9)
            ax.bar_label(rect, fmt='%.2f', padding=3, fontsize=8)

        # Standard 5-shot
        v_5_std = get_data(row, f'five_{metric_prefix}_std')
        if pd.notna(v_5_std):
            c = c_deberta if is_deberta_base else (c_smollm if is_smollm else c_5_std)
            rect = ax.barh(i - 0.5*width, v_5_std, width, color=c, alpha=0.9)
            ax.bar_label(rect, fmt='%.2f', padding=3, fontsize=8)

        # Enhanced 0-shot
        v_0_enh = get_data(row, f'zero_{metric_prefix}_enh')
        if pd.notna(v_0_enh):
            rect = ax.barh(i + 0.5*width, v_0_enh, width, color=c_0_enh, alpha=0.9)
            ax.bar_label(rect, fmt='%.2f', padding=3, fontsize=8)

        # Enhanced 5-shot
        v_5_enh = get_data(row, f'five_{metric_prefix}_enh')
        if pd.notna(v_5_enh):
            rect = ax.barh(i + 1.5*width, v_5_enh, width, color=c_5_enh, alpha=0.9)
            ax.bar_label(rect, fmt='%.2f', padding=3, fontsize=8)
            
        # Add improvement annotations if requested
        if show_improvement:
            # Standard Improvement
            if pd.notna(v_0_std) and pd.notna(v_5_std) and v_0_std > 0:
                impr = (v_5_std - v_0_std) / v_0_std * 100
                color = '#007000' if impr >= 0 else '#D00000'
                ax.text(max(v_0_std, v_5_std) + 0.05, i - 1.0*width, f'{impr:+.1f}%', 
                        va='center', ha='left', fontsize=8, fontweight='bold', color=color)
            
            # Enhanced Improvement
            if pd.notna(v_0_enh) and pd.notna(v_5_enh) and v_0_enh > 0:
                impr = (v_5_enh - v_0_enh) / v_0_enh * 100
                color = '#007000' if impr >= 0 else '#D00000'
                ax.text(max(v_0_enh, v_5_enh) + 0.05, i + 1.0*width, f'{impr:+.1f}%', 
                        va='center', ha='left', fontsize=8, fontweight='bold', color=color)

    ax.set_yticks(x)
    ax.set_yticklabels(models)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    
    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c_0_std, label='0-shot (Standard)'),
        Patch(facecolor=c_5_std, label='5-shot (Standard)'),
        Patch(facecolor=c_0_enh, label='0-shot (Enhanced)'),
        Patch(facecolor=c_5_enh, label='5-shot (Enhanced)'),
        Patch(facecolor=c_deberta, label='DeBERTa Base'),
        Patch(facecolor=c_smollm, label='SmolLM'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path.name}")

def plot_accuracy(df: pd.DataFrame, output_dir: Path):
    """Plot Accuracy for all models (Grouped)."""
    plot_grouped_bars(df, 'accuracy', 'Model Accuracy: Standard vs DeBERTa Enhanced', 'Accuracy', output_dir / 'accuracy_comparison.png')

def plot_spearman(df: pd.DataFrame, output_dir: Path):
    """Plot Spearman for all models (Grouped)."""
    plot_grouped_bars(df, 'spearman', 'Model Spearman Correlation: Standard vs DeBERTa Enhanced', 'Spearman Correlation', output_dir / 'spearman_comparison.png')

def plot_improvement(df: pd.DataFrame, output_dir: Path):
    """Plot Accuracy with Improvement % annotations."""
    # This is basically the same as plot_accuracy but with annotations enabled.
    plot_grouped_bars(df, 'accuracy', 'Few-shot Learning Impact: Standard vs Enhanced', 'Accuracy', output_dir / 'few_shot_improvement.png', show_improvement=True)

def plot_metric_consistency_superposed(df: pd.DataFrame, output_dir: Path):
    """Plot Accuracy vs Spearman correlation for ALL models (Standard + DeBERTa Enhanced)."""
    # Prepare 0-shot data
    z_df = df[['model', 'zero_accuracy', 'zero_spearman']].dropna().copy()
    z_df = z_df.rename(columns={'zero_accuracy': 'accuracy', 'zero_spearman': 'spearman'})
    
    # Prepare 5-shot data
    f_df = df[['model', 'five_accuracy', 'five_spearman']].dropna().copy()
    f_df = f_df.rename(columns={'five_accuracy': 'accuracy', 'five_spearman': 'spearman'})
    
    if z_df.empty and f_df.empty:
        print("No data available for superposed metric consistency plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Colors
    c_0 = '#a6cee3' # Light Blue
    c_5 = '#1f78b4' # Dark Blue
    c_deberta = '#ff7f00' # Orange for DeBERTa Base
    c_0_enhanced = '#cab2d6' # Light Purple
    c_5_enhanced = '#6a3d9a' # Dark Purple
    c_smollm_135 = '#FFB74D' # Lighter Orange
    c_smollm_360 = '#FFB74D' # Same Lighter Orange

    # Helper to get color
    def get_color(m, shot):
        if 'deberta-finetune' in m:
            return c_deberta
        if 'smollm-135M' in m:
            return c_smollm_135
        if 'smollm-360M' in m:
            return c_smollm_360
        if '-deberta' in m:
            return c_0_enhanced if shot == 0 else c_5_enhanced
        return c_0 if shot == 0 else c_5

    # Plot 0-shot
    colors_z = [get_color(m, 0) for m in z_df['model']]
    ax.scatter(z_df['accuracy'], z_df['spearman'], color=colors_z, s=100, alpha=0.9, edgecolors='k', label='0-shot', zorder=3)
    
    # Plot 5-shot
    colors_f = [get_color(m, 5) for m in f_df['model']]
    ax.scatter(f_df['accuracy'], f_df['spearman'], color=colors_f, s=100, alpha=0.9, edgecolors='k', marker='s', label='5-shot', zorder=3)
    # Connect points for same model
    common_models = set(z_df['model']).intersection(set(f_df['model']))
    for model in common_models:
        z_row = z_df[z_df['model'] == model].iloc[0]
        f_row = f_df[f_df['model'] == model].iloc[0]
        
        # Draw arrow
        ax.annotate("", 
                    xy=(f_row['accuracy'], f_row['spearman']), 
                    xytext=(z_row['accuracy'], z_row['spearman']),
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=15, color="gray", alpha=0.6, lw=1.5),
                    zorder=2)

    for _, row in z_df.iterrows():
        label = row['model'].replace('-deberta', '')
        if '-deberta' in row['model']:
            label += '*' # Mark enhanced models
        ax.text(row['accuracy'], row['spearman'], label, fontsize=10, alpha=1)

    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Spearman Correlation')
    ax.set_title('Metric Consistency: Standard vs DeBERTa Enhanced Models')
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='0-shot (Standard)', markerfacecolor=c_0, markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='s', color='w', label='5-shot (Standard)', markerfacecolor=c_5, markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='0-shot (Enhanced)', markerfacecolor=c_0_enhanced, markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='s', color='w', label='5-shot (Enhanced)', markerfacecolor=c_5_enhanced, markersize=10, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='DeBERTa Base', markerfacecolor=c_deberta, markersize=10, markeredgecolor='k'),
    ]
    ax.legend(handles=legend_elements)
    
    ax.grid(True, linestyle='--', alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_consistency.png')
    plt.close()
    print(f"Saved metric_consistency.png")

def plot_learning_potential(df: pd.DataFrame, output_dir: Path):
    """Plot Zero-shot Accuracy vs Accuracy Improvement %."""
    plot_df = df[['model', 'zero_accuracy', 'five_accuracy', 'accuracy_impr_pct']].dropna().copy()
    
    if plot_df.empty:
        print("No data available for learning potential plot.")
        return

    # Identify Standard vs Enhanced
    plot_df['base_name'] = plot_df['model'].apply(lambda x: x.replace('-deberta', ''))
    plot_df['is_enhanced'] = plot_df['model'].apply(lambda x: '-deberta' in x)

    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Colors
    c_std = '#1f78b4' # Blue (Standard)
    c_enh = '#6a3d9a' # Purple (Enhanced)
    c_deberta = '#ff7f00' # Orange (DeBERTa)
    c_pos = '#55a868' # Green (Positive Improvement)
    c_neg = '#c44e52' # Red (Negative Improvement)
    
    # Plot points and 0->5 arrows
    for idx, row in plot_df.iterrows():
        # Determine point color
        if 'deberta-finetune' in row['model']:
            color = c_deberta
        elif row['is_enhanced']:
            color = c_enh
        else:
            color = c_std
            
        # Determine arrow color based on improvement
        impr = row['accuracy_impr_pct']
        arrow_color = c_pos if impr >= 0 else c_neg
        
        # Plot point
        ax.scatter(row['zero_accuracy'], row['accuracy_impr_pct'], color=color, s=120, alpha=0.9, edgecolors='w', zorder=3)
        
        # Draw arrow from 0-shot to 5-shot (X-axis shift)
        ax.annotate("", 
                    xy=(row['five_accuracy'], row['accuracy_impr_pct']), 
                    xytext=(row['zero_accuracy'], row['accuracy_impr_pct']),
                    arrowprops=dict(arrowstyle="-|>", color=arrow_color, alpha=0.6, lw=2),
                    zorder=2)
        
        # Label point
        label = row['model'].replace('-deberta', '')
        if row['is_enhanced']:
            label += '*'
        ax.text(row['zero_accuracy'], row['accuracy_impr_pct'] + 0.8, label, fontsize=9, ha='center', va='bottom', alpha=0.8)

    # Connect Standard to Enhanced (New feature)
    # Group by base_name
    for base_name, group in plot_df.groupby('base_name'):
        if len(group) == 2:
            # We have both Standard and Enhanced
            std = group[~group['is_enhanced']].iloc[0]
            enh = group[group['is_enhanced']].iloc[0]
            
            # Draw arrow from Standard to Enhanced
            ax.annotate("",
                        xy=(enh['zero_accuracy'], enh['accuracy_impr_pct']),
                        xytext=(std['zero_accuracy'], std['accuracy_impr_pct']),
                        arrowprops=dict(arrowstyle="->", color='gray', linestyle='--', alpha=0.5, lw=1.5),
                        zorder=1)

    # Set x-axis limits
    all_scores = pd.concat([plot_df['zero_accuracy'], plot_df['five_accuracy']])
    
    # Check for DeBERTa to include in limits
    deberta_row = df[df['model'].str.contains('deberta', case=False)]
    if not deberta_row.empty:
         val = deberta_row.iloc[0]['zero_accuracy']
         if pd.notna(val):
             all_scores = pd.concat([all_scores, pd.Series([val])])

    x_min, x_max = all_scores.min(), all_scores.max()
    padding = (x_max - x_min) * 0.1
    ax.set_xlim(x_min - padding, x_max + padding)

    ax.set_xlabel('Baseline Performance (Zero-shot Accuracy)')
    ax.set_ylabel('Benefit from Few-shot (Accuracy Improvement %)')
    ax.set_title('Learning Potential: Baseline vs Improvement (Accuracy)')
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3) # Zero improvement line
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add DeBERTa vertical line
    if not deberta_row.empty:
        deberta_val = deberta_row.iloc[0]['zero_accuracy']
        if pd.notna(deberta_val):
            ax.axvline(deberta_val, color=c_deberta, linestyle='--', linewidth=2, alpha=0.8, label='DeBERTa Base')
            ylim = ax.get_ylim()
            ax.text(deberta_val, ylim[1] - (ylim[1]-ylim[0])*0.05, ' DeBERTa Base', color=c_deberta, fontweight='bold', ha='left', va='top', fontsize=9)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Standard Model', markerfacecolor=c_std, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Enhanced Model', markerfacecolor=c_enh, markersize=10),
        Line2D([0], [0], color=c_pos, lw=2, label='Positive Improvement'),
        Line2D([0], [0], color=c_neg, lw=2, label='Negative Improvement'),
        Line2D([0], [0], color='gray', linestyle='--', lw=1.5, label='Std -> Enh Shift'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / 'learning_potential.png')
    plt.close()
    print(f"Saved learning_potential.png")

def plot_learning_potential_spearman(df: pd.DataFrame, output_dir: Path):
    """Plot Zero-shot Spearman vs Spearman Improvement %."""
    plot_df = df[['model', 'zero_spearman', 'five_spearman', 'spearman_impr_pct']].dropna().copy()
    
    if plot_df.empty:
        print("No data available for learning potential (Spearman) plot.")
        return

    # Identify Standard vs Enhanced
    plot_df['base_name'] = plot_df['model'].apply(lambda x: x.replace('-deberta', ''))
    plot_df['is_enhanced'] = plot_df['model'].apply(lambda x: '-deberta' in x)

    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Colors
    c_std = '#1f78b4' # Blue (Standard)
    c_enh = '#6a3d9a' # Purple (Enhanced)
    c_deberta = '#ff7f00' # Orange (DeBERTa)
    c_pos = '#55a868' # Green (Positive Improvement)
    c_neg = '#c44e52' # Red (Negative Improvement)
    
    # Plot points and 0->5 arrows
    for idx, row in plot_df.iterrows():
        # Determine point color
        if 'deberta-finetune' in row['model']:
            color = c_deberta
        elif row['is_enhanced']:
            color = c_enh
        else:
            color = c_std
            
        # Determine arrow color based on improvement
        impr = row['spearman_impr_pct']
        arrow_color = c_pos if impr >= 0 else c_neg
        
        # Plot point
        ax.scatter(row['zero_spearman'], row['spearman_impr_pct'], color=color, s=120, alpha=0.9, edgecolors='w', zorder=3)
        
        # Draw arrow from 0-shot to 5-shot (X-axis shift)
        ax.annotate("", 
                    xy=(row['five_spearman'], row['spearman_impr_pct']), 
                    xytext=(row['zero_spearman'], row['spearman_impr_pct']),
                    arrowprops=dict(arrowstyle="-|>", color=arrow_color, alpha=0.6, lw=2),
                    zorder=2)
        
        # Label point
        label = row['model'].replace('-deberta', '')
        if row['is_enhanced']:
            label += '*'
        ax.text(row['zero_spearman'], row['spearman_impr_pct'] + 0.8, label, fontsize=9, ha='center', va='bottom', alpha=0.8)

    # Connect Standard to Enhanced (New feature)
    for base_name, group in plot_df.groupby('base_name'):
        if len(group) == 2:
            std = group[~group['is_enhanced']].iloc[0]
            enh = group[group['is_enhanced']].iloc[0]
            
            # Draw arrow from Standard to Enhanced
            ax.annotate("",
                        xy=(enh['zero_spearman'], enh['spearman_impr_pct']),
                        xytext=(std['zero_spearman'], std['spearman_impr_pct']),
                        arrowprops=dict(arrowstyle="->", color='gray', linestyle='--', alpha=0.5, lw=1.5),
                        zorder=1)

    # Set x-axis limits
    all_scores = pd.concat([plot_df['zero_spearman'], plot_df['five_spearman']])
    
    # Check for DeBERTa to include in limits
    deberta_row = df[df['model'].str.contains('deberta', case=False)]
    if not deberta_row.empty:
         val = deberta_row.iloc[0]['zero_spearman']
         if pd.notna(val):
             all_scores = pd.concat([all_scores, pd.Series([val])])

    x_min, x_max = all_scores.min(), all_scores.max()
    padding = (x_max - x_min) * 0.1
    ax.set_xlim(x_min - padding, x_max + padding)

    ax.set_xlabel('Baseline Performance (Zero-shot Spearman)')
    ax.set_ylabel('Benefit from Few-shot (Spearman Improvement %)')
    ax.set_title('Learning Potential: Baseline vs Improvement (Spearman)')
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3) # Zero improvement line
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add DeBERTa vertical line
    if not deberta_row.empty:
        deberta_val = deberta_row.iloc[0]['zero_spearman']
        if pd.notna(deberta_val):
            ax.axvline(deberta_val, color=c_deberta, linestyle='--', linewidth=2, alpha=0.8, label='DeBERTa Base')
            ylim = ax.get_ylim()
            ax.text(deberta_val, ylim[1] - (ylim[1]-ylim[0])*0.05, ' DeBERTa Base', color=c_deberta, fontweight='bold', ha='left', va='top', fontsize=9)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Standard Model', markerfacecolor=c_std, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Enhanced Model', markerfacecolor=c_enh, markersize=10),
        Line2D([0], [0], color=c_pos, lw=2, label='Positive Improvement'),
        Line2D([0], [0], color=c_neg, lw=2, label='Negative Improvement'),
        Line2D([0], [0], color='gray', linestyle='--', lw=1.5, label='Std -> Enh Shift'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_potential_spearman.png')
    plt.close()
    print(f"Saved learning_potential_spearman.png")

def main():
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    results_dir = repo_root / "results"
    csv_path = results_dir / "summary_0shot_5shot_scores.csv"
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
        
    print(f"Reading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    set_style()
    
    plot_accuracy(df, results_dir)
    plot_spearman(df, results_dir)
    plot_improvement(df, results_dir)
    plot_metric_consistency_superposed(df, results_dir)
    plot_learning_potential(df, results_dir)
    plot_learning_potential_spearman(df, results_dir)
    
    print("All plots generated successfully.")

if __name__ == "__main__":
    main()
