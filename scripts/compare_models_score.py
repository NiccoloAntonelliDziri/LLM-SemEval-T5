"""
Script to compare LLM and DeBERTa model scores on the first 100 elements.
Scoring is done using the scoring.py functions and results are saved to a CSV file.
"""

import sys
import json
import os
import csv
import subprocess
from pathlib import Path

def load_first_n_predictions(filepath, n=100):
    """
    Load predictions for the first n items (IDs 0 to n-1).
    If a prediction is valid (compatible range), it is kept.
    If an ID is missing, we fill it with a default value of 3.
    """
    predictions_map = {}
    
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if 'id' in item and 'prediction' in item:
                            # Handle string or int ids
                            try:
                                item_id = int(item['id'])
                            except ValueError:
                                continue
                                
                            if 0 <= item_id < n:
                                predictions_map[item_id] = item['prediction']
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Error reading {filepath}: {e}")
    
    # Construct the final list, imputing missing values
    final_predictions = []
    missing_count = 0
    
    for i in range(n):
        if i in predictions_map:
            final_predictions.append({"id": str(i), "prediction": predictions_map[i]})
        else:
            # Impute missing value
            final_predictions.append({"id": str(i), "prediction": 3.0})
            missing_count += 1
            
    if missing_count > 0:
        print(f"    (Imputed {missing_count} missing predictions with value 3.0)")
        
    return final_predictions

def load_gold_data(filepath):
    """Load all gold data from reference file."""
    gold_data = []
    with open(filepath, 'r') as f:
        for line in f:
            gold_data.append(json.loads(line))
    return gold_data

def filter_first_n_items(data, n=100):
    """Filter data to only first n items by ID."""
    return [item for item in data if int(item.get('id', -1)) < n]

def calculate_scores(predictions_data, temp_ref_file, temp_pred_file, temp_score_file, scoring_script_path):
    """Calculate spearman and accuracy scores by calling scoring.py."""
    try:
        # Write temporary predictions file
        with open(temp_pred_file, 'w') as f:
            for pred in predictions_data:
                f.write(json.dumps(pred) + '\n')
        
        # Call scoring.py script
        # Usage: python3 scoring.py ref_filepath pred_filepath output_filepath
        subprocess.run(
            [sys.executable, scoring_script_path, temp_ref_file, temp_pred_file, temp_score_file],
            check=True,
            capture_output=True
        )

        # Read results
        if os.path.exists(temp_score_file):
            with open(temp_score_file, 'r') as f:
                scores = json.load(f)
                return scores.get('spearman'), scores.get('accuracy')
        return None, None

    except subprocess.CalledProcessError as e:
        print(f"Error running scoring script: {e.stderr.decode()}")
        return None, None
    except Exception as e:
        print(f"Error calculating scores: {e}")
        return None, None
    finally:
        # Clean up temporary files
        if os.path.exists(temp_pred_file):
            os.remove(temp_pred_file)
        if os.path.exists(temp_score_file):
            os.remove(temp_score_file)

def score_deberta_models(base_path, temp_ref_file, output_results, scoring_script_path):
    """Score DeBERTa model predictions."""
    print("\n=== Scoring DeBERTa Models ===")
    
    deberta_dirs = [
        'deberta-finetune',
        'deberta-finetune-2',
        'deberta-finetune-3',
        'deberta-refinement',
        'smollm-finetune-135M',
        'smollm-finetune-360M'
    ]
    
    temp_pred_file = os.path.join(base_path, 'temp_predictions.jsonl')
    temp_score_file = os.path.join(base_path, 'temp_score.json')

    for dir_name in deberta_dirs:
        dir_path = os.path.join(base_path, dir_name)
        pred_file = os.path.join(dir_path, 'predictions.jsonl')
        
        if os.path.exists(pred_file):
            print(f"\nScoring {dir_name}...")
            predictions = load_first_n_predictions(pred_file, n=100)
            
            if predictions:
                spearman, accuracy = calculate_scores(predictions, temp_ref_file, temp_pred_file, temp_score_file, scoring_script_path)
                
                if spearman is not None and accuracy is not None:
                    output_results.append({
                        'model_type': 'DeBERTa',
                        'model_name': dir_name,
                        'category': '-',
                        'spearman': spearman,
                        'accuracy': accuracy
                    })
                    print(f"  Spearman: {spearman}, Accuracy: {accuracy}")
        else:
            print(f"  Predictions file not found: {pred_file}")

def score_llm_models(base_path, temp_ref_file, output_results, scoring_script_path):
    """Score LLM model predictions."""
    print("\n=== Scoring LLM Models ===")
    
    llm_base_path = os.path.join(base_path, 'llm-ollama')
    temp_pred_file = os.path.join(base_path, 'temp_predictions.jsonl')
    temp_score_file = os.path.join(base_path, 'temp_score.json')
    
    # Scan for all LLM model directories
    for category in os.listdir(llm_base_path):
        category_path = os.path.join(llm_base_path, category)
        if not os.path.isdir(category_path):
            continue
        
        for model_name in os.listdir(category_path):
            model_path = os.path.join(category_path, model_name)
            if not os.path.isdir(model_path):
                continue
            
            pred_file = os.path.join(model_path, 'predictions.jsonl')
            
            if os.path.exists(pred_file):
                print(f"\nScoring {category}/{model_name}...")
                predictions = load_first_n_predictions(pred_file, n=100)
                
                if predictions:
                    spearman, accuracy = calculate_scores(predictions, temp_ref_file, temp_pred_file, temp_score_file, scoring_script_path)
                    
                    if spearman is not None and accuracy is not None:
                        output_results.append({
                            'model_type': 'LLM',
                            'model_name': model_name,
                            'category': category,
                            'spearman': spearman,
                            'accuracy': accuracy
                        })
                        print(f"  Spearman: {spearman}, Accuracy: {accuracy}")

def save_results_to_csv(results, output_dir, output_filename='model_comparison_first100.csv'):
    """Save results to CSV file."""
    output_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if results:
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['model_type', 'model_name', 'category', 'spearman', 'accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✓ Results saved to: {output_path}")
        return output_path
    else:
        print("\nNo results to save.")
        return None

def main():
    base_path = '/home/niccolo/Torino/LLM-SamEval-T5'
    scoring_script_path = os.path.join(base_path, 'score/scoring.py')
    ref_file = os.path.join(base_path, 'deberta-finetune-2/ref.jsonl')
    output_dir = os.path.join(base_path, 'results')
    
    print("Loading reference data...")
    gold_data = load_gold_data(ref_file)
    # Filter to first 100 items
    gold_data = filter_first_n_items(gold_data, n=100)
    print(f"Loaded {len(gold_data)} reference items")
    
    # Create temporary reference file for the subset
    temp_ref_file = os.path.join(base_path, 'temp_ref.jsonl')
    try:
        with open(temp_ref_file, 'w') as f:
            for item in gold_data:
                f.write(json.dumps(item) + '\n')
    
        output_results = []
        
        # Score all models
        score_deberta_models(base_path, temp_ref_file, output_results, scoring_script_path)
        score_llm_models(base_path, temp_ref_file, output_results, scoring_script_path)
        
        # Save results to CSV
        save_results_to_csv(output_results, output_dir)
        
        # Print summary
        if output_results:
            print("\n=== Summary ===")
            print(f"Total models scored: {len(output_results)}")
            
            deberta_models = [r for r in output_results if r['model_type'] == 'DeBERTa']
            llm_models = [r for r in output_results if r['model_type'] == 'LLM']
            
            print(f"DeBERTa models: {len(deberta_models)}")
            print(f"LLM models: {len(llm_models)}")
            
            if deberta_models:
                avg_deberta_spearman = sum(r['spearman'] for r in deberta_models) / len(deberta_models)
                avg_deberta_accuracy = sum(r['accuracy'] for r in deberta_models) / len(deberta_models)
                print(f"\nDeBERTa Average Spearman: {avg_deberta_spearman:.4f}")
                print(f"DeBERTa Average Accuracy: {avg_deberta_accuracy:.4f}")
            
            if llm_models:
                avg_llm_spearman = sum(r['spearman'] for r in llm_models) / len(llm_models)
                avg_llm_accuracy = sum(r['accuracy'] for r in llm_models) / len(llm_models)
                print(f"\nLLM Average Spearman: {avg_llm_spearman:.4f}")
                print(f"LLM Average Accuracy: {avg_llm_accuracy:.4f}")
    
    finally:
        # Cleanup temp ref file
        if os.path.exists(temp_ref_file):
            os.remove(temp_ref_file)

if __name__ == '__main__':
    main()
