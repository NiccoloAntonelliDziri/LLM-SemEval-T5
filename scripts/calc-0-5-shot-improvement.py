#!/usr/bin/env python3

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

@dataclass
class Scores:
	accuracy: Optional[float]
	spearman: Optional[float]
	avg_time: Optional[float] = None

def read_timing_file(timing_path: Path) -> Optional[float]:
	try:
		with timing_path.open("r", encoding="utf-8") as f:
			for line in f:
				if line.startswith("avg_time_sec:"):
					return float(line.split(":")[1].strip())
		return None
	except FileNotFoundError:
		return None
	except Exception as e:
		print(f"Warning: failed to read {timing_path}: {e}")
		return None

def read_score_file(score_path: Path) -> Optional[Scores]:
	try:
		with score_path.open("r", encoding="utf-8") as f:
			data = json.load(f)
		return Scores(
			accuracy=float(data.get("accuracy")) if data.get("accuracy") is not None else None,
			spearman=float(data.get("spearman")) if data.get("spearman") is not None else None,
		)
	except FileNotFoundError:
		return None
	except Exception as e:
		print(f"Warning: failed to read {score_path}: {e}")
		return None


def pct_improvement(baseline: Optional[float], new: Optional[float]) -> Optional[float]:
	if baseline is None or new is None:
		return None
	if baseline == 0:
		return None
	return (new - baseline) / baseline * 100.0


def collect_scores(ollama_dir: Path) -> Dict[str, Dict[str, Scores]]:
	zero_dir = ollama_dir / "zero-shot"
	five_dir = ollama_dir / "five-shot"

	results: Dict[str, Dict[str, Scores]] = {}

	def load_split(split_dir: Path, split_name: str) -> None:
		if not split_dir.exists():
			return
		for model_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
			score_path = model_dir / "score.json"
			scores = read_score_file(score_path)
			if scores is None:
				continue
			
			timing_path = model_dir / "timing.txt"
			scores.avg_time = read_timing_file(timing_path)

			rec = results.setdefault(model_dir.name, {})
			rec[split_name] = scores

	load_split(zero_dir, "zero")
	load_split(five_dir, "five")
	return results


def write_csv_report(results: Dict[str, Dict[str, Scores]], out_path: Path) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	headers = [
		"model",
		"zero_accuracy",
		"five_accuracy",
		"accuracy_impr_pct",
		"zero_spearman",
		"five_spearman",
		"spearman_impr_pct",
		"zero_avg_time",
		"five_avg_time",
	]
	with out_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(headers)
		for model, splits in sorted(results.items()):
			zero = splits.get("zero")
			five = splits.get("five")

			zero_acc = zero.accuracy if zero else None
			five_acc = five.accuracy if five else None
			zero_spr = zero.spearman if zero else None
			five_spr = five.spearman if five else None
			zero_time = zero.avg_time if zero else None
			five_time = five.avg_time if five else None

			acc_impr = pct_improvement(zero_acc, five_acc)
			spr_impr = pct_improvement(zero_spr, five_spr)

			def fmt(x: Optional[float]) -> str:
				if x is None:
					return ""
				return f"{x:.6f}"

			def fmt_pct(x: Optional[float]) -> str:
				if x is None:
					return ""
				return f"{x:.2f}"

			row = [
				model,
				fmt(zero_acc),
				fmt(five_acc),
				fmt_pct(acc_impr),
				fmt(zero_spr),
				fmt(five_spr),
				fmt_pct(spr_impr),
				fmt(zero_time),
				fmt(five_time),
			]
			writer.writerow(row)


def main() -> None:
	# repo_root = scripts/.. (this file resides in scripts/)
	script_path = Path(__file__).resolve()
	repo_root = script_path.parents[1]
	ollama_dir = repo_root / "llm-ollama"

	if not ollama_dir.exists():
		raise SystemExit(f"Directory not found: {ollama_dir}")

	results = collect_scores(ollama_dir)
	
	# Add DeBERTa score
	deberta_dir = repo_root / "deberta-finetune-2" # Because the second version is a little better
	deberta_score_path = deberta_dir / "score.json"
	deberta_scores = read_score_file(deberta_score_path)
	if deberta_scores:
		# We'll put it under "zero" split so it appears in the first columns
		results["deberta-finetune"] = {"zero": deberta_scores}

	if not results:
		raise SystemExit("No scores found under llm-ollama/{zero-shot,five-shot} or deberta-finetune")

	results_dir = repo_root / "results"
	out_csv = results_dir / "summary_0shot_5shot_scores.csv"
	write_csv_report(results, out_csv)
	print(f"Wrote summary: {out_csv}")


if __name__ == "__main__":
	main()

