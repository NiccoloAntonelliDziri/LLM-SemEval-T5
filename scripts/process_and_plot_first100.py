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

def process_data(input_file):
    """Process the long-format CSV into the wide-format summary CSV."""
    df = pd.read_csv(input_file)
    
    # Dictionary to store processed rows: model_name -> dict of metrics
    processed_data = {}
    
    for _, row in df.iterrows():
        model_type = row['model_type']
        model_name = row['model_name']
        category = row['category']
        spearman = row['spearman']
        accuracy = row['accuracy']
        
        # Determine the target row name (model identifier) and metric prefix
        target_model_name = model_name
        metric_prefix = None # 'zero' or 'five'
        
        if model_type == 'DeBERTa':
            # DeBERTa models are treated as having "zero" metrics for plotting purposes
            # (or just simple values, but standardizing on zero_* allows reuse of plot code)
            target_model_name = model_name
            metric_prefix = 'zero'
        
        elif model_type == 'LLM':
            if category == 'zero-shot':
                target_model_name = model_name
                metric_prefix = 'zero'
            elif category == 'five-shot':
                target_model_name = model_name
                metric_prefix = 'five'
            elif category == 'zero-shot-deberta':
                target_model_name = f"{model_name}-deberta"
                metric_prefix = 'zero'
            elif category == 'five-shot-deberta':
                target_model_name = f"{model_name}-deberta"
                metric_prefix = 'five'
        
        if metric_prefix:
            if target_model_name not in processed_data:
                processed_data[target_model_name] = {'model': target_model_name}
            
            processed_data[target_model_name][f'{metric_prefix}_accuracy'] = accuracy
            processed_data[target_model_name][f'{metric_prefix}_spearman'] = spearman

    # Convert to DataFrame
    summary_df = pd.DataFrame(list(processed_data.values()))
    
    # Calculate improvements
    if 'zero_accuracy' in summary_df.columns and 'five_accuracy' in summary_df.columns:
        summary_df['accuracy_impr_pct'] = (summary_df['five_accuracy'] - summary_df['zero_accuracy']) / summary_df['zero_accuracy'] * 100
    else:
        summary_df['accuracy_impr_pct'] = None
        
    if 'zero_spearman' in summary_df.columns and 'five_spearman' in summary_df.columns:
        summary_df['spearman_impr_pct'] = (summary_df['five_spearman'] - summary_df['zero_spearman']) / summary_df['zero_spearman'] * 100
    else:
        summary_df['spearman_impr_pct'] = None
        
    # Add empty time columns to match schema if necessary (though plot_results doesn't seem to strictly require them for plotting)
    summary_df['zero_avg_time'] = None
    summary_df['five_avg_time'] = None
    
    return summary_df

def main():
    base_path = Path('/home/niccolo/Torino/LLM-SamEval-T5')
    results_dir = base_path / 'results'
    input_file = results_dir / 'model_comparison_first100.csv'
    output_csv = results_dir / 'summary_first100.csv'
    output_plots_dir = results_dir / 'plots_first100'
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found.")
        return

    print(f"Processing {input_file}...")
    summary_df = process_data(input_file)
    
    print(f"Saving summary to {output_csv}...")
    summary_df.to_csv(output_csv, index=False)
    
    # Create output directory for plots
    output_plots_dir.mkdir(exist_ok=True)
    
    print(f"Generating plots in {output_plots_dir}...")
    
    # Set style
    plot_results.set_style()
    
    # Generate plots using imported functions
    try:
        plot_results.plot_accuracy(summary_df, output_plots_dir)
        plot_results.plot_spearman(summary_df, output_plots_dir)
        plot_results.plot_improvement(summary_df, output_plots_dir)
        plot_results.plot_metric_consistency_superposed(summary_df, output_plots_dir)
        plot_results.plot_learning_potential(summary_df, output_plots_dir)
        # Note: plot_learning_potential_spearman might crash if missing columns, let's check
        if hasattr(plot_results, 'plot_learning_potential_spearman'):
             plot_results.plot_learning_potential_spearman(summary_df, output_plots_dir)
             
        print("All plots generated successfully.")
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
