Task website: https://nlu-lab.github.io/semeval.html

## Requirements

- Python 3.13.11
- Pytorch for your system
- Ollama (install from https://github.com/ollama/ollama)
- Other requirements: `pip install -r requirements.txt`

## Repo structure

A simple overview of the main files and folders in this repository:

- `data/` — dataset files used for training and evaluation (from https://github.com/Janosch-Gehring/ambistory )
- `deberta-finetune/` — results of deberta finetuned model
- `llm-prompting/` — results of LLMs
- `score/` — scoring utilities used to evaluate predictions (from https://github.com/Janosch-Gehring/semeval26-05-scripts) 
- `scripts/` — notebooks for running models and experiments (examples use Ollama and DeBERTa).
- `requirements.txt` — Python dependencies to install. (in addition to pytorch)
