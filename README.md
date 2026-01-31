Task website: https://nlu-lab.github.io/semeval.html

## Requirements

- Python 3.13.11
- Pytorch for your system
- Ollama (install from https://github.com/ollama/ollama)
- Other requirements: `pip install -r requirements.txt`

## Repo structure

A simple overview of the main files and folders in this repository:

- `data/` — dataset files used for training and evaluation (from https://github.com/Janosch-Gehring/ambistory )
- `DeBERTa-NLI/` — results of the fine-tuned DeBERTa model used for enhancing LLM predictions
- `llm-ollama/` — results of LLM zero-shot and five-shot prompting
- `score/` — scoring utilities used to evaluate predictions (from https://github.com/Janosch-Gehring/semeval26-05-scripts) 
- `scripts/` — notebooks for running models and experiments (examples use Ollama and DeBERTa).
- `requirements.txt` — Python dependencies to install. (in addition to pytorch)
- `results/` — generated plots and summary CSV files.
- `report/` — contains the final report of the project.

## Results

### Metric Consistency
![Metric Consistency](results/plots_unfiltered/metric_consistency.png)
