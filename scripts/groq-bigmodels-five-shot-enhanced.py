import os
import json
import time
from tqdm import tqdm
from groq import Groq
import sys
import subprocess

# Configuration
MODEL_NAME = "openai/gpt-oss-120b"
DATA_PATH = "../data/dev.json"
TRAIN_DATA_PATH = "../data/train.json"
OUTPUT_DIR = "../llm-ollama/five-shot-deberta/openai-gpt-oss-120b"
DEBERTA_PRED_FILE = "../deberta-finetune-2/predictions.jsonl"
DEBERTA_SCORE_FILE = "../deberta-finetune-2/score.json"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def load_deberta_results(pred_file, score_file):
    with open(score_file, 'r') as f:
        scores = json.load(f)
    
    preds = {}
    with open(pred_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            preds[str(item['id'])] = item['prediction']
            
    return scores, preds

# Combined prompt as per Groq reasoning recommendations (avoid system prompts)
FULL_PROMPT_TEMPLATE = (
    """
    You are an expert NLU annotator. Your job is to rate how plausible a candidate meaning (sense)
    is for the HOMONYM used in the target sentence within the short story.

    Return ONLY a single JSON object with one key: "score" and an integer value 1, 2, 3, 4 or 5.
    Integer mapping:
      1 = Definitely not
      2 = Probably not
      3 = Ambiguous / Unsure
      4 = Probably yes
      5 = Definitely yes

    The response must be a JSON object and nothing else, for example: {{"score": 4}}

    [EXAMPLES]
    {few_shot_examples}

    Now, label this new instance:

    [STORY]
    {full_story_text}

    [HOMONYM]
    {homonym}

    [CANDIDATE SENSE]
    {sense_text}

    [ADDITIONAL CONTEXT]
    A DeBERTa model (Accuracy: {deberta_accuracy:.2f}, Spearman Correlation: {deberta_spearman:.2f}) predicted a score of {deberta_prediction:.2f} for this example.
    You can use this information to guide your decision, but rely on your own judgment if the context strongly suggests otherwise.

    [TASK]
    Based on the STORY above, decide how plausible it is that the HOMONYM is used with the
    CANDIDATE SENSE in the target sentence.

    Return ONLY a single JSON object with one key "score" and an integer value (1-5)
    as described by the system message. Example output: {{"score": 3}}
    """
)

def create_full_story_text(item):
    fullstory = f"{item.get('precontext', '')} {item.get('sentence', '')} {item.get('ending', '')}"
    return fullstory.strip()

def build_few_shot_examples(data):
    """Build a five-shot examples string (one example per score 1-5) from `data`.

    Uses rounded average to determine the representative score for an item.
    """
    examples = []
    used_ids = set()

    for score in range(1, 6):
        found = False
        for idx, row in data.items():
            if str(idx) in used_ids:
                continue
            try:
                avg = float(row.get("average", 0))
            except Exception:
                avg = 0.0
            if round(avg) == score:
                full = create_full_story_text(row)
                sense = f"{row.get('judged_meaning', '')} as in \"{row.get('example_sentence','')}\""
                ex = (
                    "[STORY]\n"
                    f"{full}\n\n"
                    "[HOMONYM]\n"
                    f"{row.get('homonym', '')}\n\n"
                    "[CANDIDATE SENSE]\n"
                    f"{sense}\n\n"
                    "[EXAMPLE OUTPUT]\n"
                    f'{{"score": {score}}}'
                )
                examples.append(ex)
                used_ids.add(str(idx))
                found = True
                break
        if not found:
             # fallback: include a short template example if no example found for this score
            examples.append(f"[EXAMPLE OUTPUT]\n{{\"score\": {score}}}")

    return "\n\n".join(examples)

# Load data
print(f"Loading data from {DATA_PATH}...")
with open(DATA_PATH, "r") as f:
    data = json.load(f)

print(f"Loading training data from {TRAIN_DATA_PATH}...")
with open(TRAIN_DATA_PATH, "r") as f:
    train_data = json.load(f)

# Load DeBERTa results
print(f"Loading DeBERTa results from {DEBERTA_PRED_FILE}...")
deberta_scores, deberta_preds = load_deberta_results(DEBERTA_PRED_FILE, DEBERTA_SCORE_FILE)

# Build few-shot examples once
few_shot_examples_text = build_few_shot_examples(train_data)

def create_message(item, deberta_pred, deberta_scores):
    sense = f"{item.get('judged_meaning', '')} as in \"{item.get('example_sentence', '')}\"".strip()
    homonym = item.get("homonym", "")
    full_story_text = create_full_story_text(item)
    
    return FULL_PROMPT_TEMPLATE.format(
        few_shot_examples=few_shot_examples_text,
        full_story_text=full_story_text,
        homonym=homonym,
        sense_text=sense,
        deberta_accuracy=deberta_scores['accuracy'],
        deberta_spearman=deberta_scores['spearman'],
        deberta_prediction=deberta_pred
    )

# Prepare output files
pred_file = os.path.join(OUTPUT_DIR, "predictions.jsonl")
ref_file = os.path.join(OUTPUT_DIR, "ref.jsonl")
failed_file = os.path.join(OUTPUT_DIR, "failed_ids.jsonl")

# Check if we can resume
processed_ids = set()
if os.path.exists(pred_file):
    with open(pred_file, "r") as f:
        for line in f:
            try:
                processed_ids.add(json.loads(line)["id"])
            except:
                pass
    print(f"Resuming from {len(processed_ids)} processed items.")

# Open files for appending
with open(pred_file, "a") as f_pred, open(ref_file, "a") as f_ref, open(failed_file, "a") as f_fail:
    
    # Iterate over data
    for item_id, item in tqdm(data.items()):
        if item_id in processed_ids:
            continue
            
        deberta_pred = deberta_preds.get(str(item_id), 0.0) # Default to 0? Or maybe 3? Or check if missing
        
        user_content = create_message(item, deberta_pred, deberta_scores)
        
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": user_content,
                    }
                ],
                model=MODEL_NAME,
                temperature=0, # Deterministic
                reasoning_effort="low", # Minimize token usage for rate limits
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "homonym_score",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "score": {
                                    "type": "integer",
                                    "enum": [1, 2, 3, 4, 5]
                                }
                            },
                            "required": ["score"],
                            "additionalProperties": False
                        }
                    }
                },
            )
            
            response_content = chat_completion.choices[0].message.content
            
            # Parse JSON
            try:
                response_json = json.loads(response_content)
                score = response_json.get("score")
                
                if score is not None:
                    # Save prediction
                    f_pred.write(json.dumps({"id": item_id, "prediction": score}) + "\n")
                    f_pred.flush()
                    
                    # Save reference (gold)
                    f_ref.write(json.dumps({"id": item_id, "label": item["choices"]}) + "\n")
                    f_ref.flush()
                else:
                    raise ValueError("No score found in response")
                    
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for {item_id}: {response_content}")
                f_fail.write(json.dumps({"id": item_id, "error": "json_parse", "content": response_content}) + "\n")
                f_fail.flush()
                
        except Exception as e:
            print(f"Error processing {item_id}: {e}")
            f_fail.write(json.dumps({"id": item_id, "error": str(e)}) + "\n")
            f_fail.flush()
            # Sleep a bit on error to avoid hammering if it's a rate limit
            time.sleep(2)

print("Done!")

# Run scoring script
score_file = os.path.join(OUTPUT_DIR, "score.json")
scoring_script = "../score/scoring.py"

print(f"Running scoring script: {scoring_script}")
subprocess.run([sys.executable, scoring_script, ref_file, pred_file, score_file])
