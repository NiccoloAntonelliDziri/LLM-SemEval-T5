#!/usr/bin/env python3

import pandas as pd
import sys
import os
from pathlib import Path

# Add the scripts directory to path to allow importing plot_results
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import plot_results
except ImportError:
    # If generic import fails (e.g. if script is run from root), try relative
    sys.path.append('scripts')
    import plot_results

def main():
    base_path = Path('/home/niccolo/Torino/LLM-SamEval-T5')
    results_dir = base_path / 'results'
    input_file = results_dir / 'summary_0shot_5shot_scores.csv'
    output_plots_dir = results_dir / 'plots_summary_filtered'
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found.")
        return

    print(f"Reading data from {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Original dataframe size: {len(df)}")
    
    # Filter out models containing 'think'
    # "plots for all of the models that are not run on only 100 elements ie the thinking models"
    # This means we exclude the thinking models.
    filtered_df = df[~df['model'].str.contains('think', case=False, na=False)].copy()
    
    # Also double check expected thinking models from list: olmo-3-think, deepseek-r1-think, gpt-oss-20b-think
    # The string match 'think' should catch them all.
    
    print(f"Filtered dataframe size: {len(filtered_df)}")
    print("Excluded models:")
    excluded = df[df['model'].str.contains('think', case=False, na=False)]['model'].tolist()
    for m in excluded:
        print(f" - {m}")
    
    # Create output directory for plots
    output_plots_dir.mkdir(exist_ok=True)
    
    print(f"Generating plots in {output_plots_dir}...")
    
    # Set style
    plot_results.set_style()
    
    # Generate plots using imported functions
    try:
        plot_results.plot_accuracy(filtered_df, output_plots_dir)
        plot_results.plot_spearman(filtered_df, output_plots_dir)
        plot_results.plot_improvement(filtered_df, output_plots_dir)
        plot_results.plot_metric_consistency_superposed(filtered_df, output_plots_dir)
        plot_results.plot_learning_potential(filtered_df, output_plots_dir)
        # Note: plot_learning_potential_spearman might crash if missing columns, let's check
        if hasattr(plot_results, 'plot_learning_potential_spearman'):
             plot_results.plot_learning_potential_spearman(filtered_df, output_plots_dir)
             
        print("All plots generated successfully.")
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
