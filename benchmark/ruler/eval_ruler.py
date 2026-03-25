"""
RULER benchmark evaluation for OmniKV.
Loads pre-generated RULER data (with chat template baked in) and evaluates using OmniKV model.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from tqdm import tqdm

# Add OmniKV root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformers import GenerationConfig
from infer import get_any_chat_api
from tiny_tools.read_json import read_config
from benchmark.ruler.ruler_metrics import needle_score, multi_words, multi_number, string_match_part


TASK_CONFIG = {
    'niah_single_1':  {'gen_len': 64,  'metric': needle_score,      'metric_name': 'needle_score'},
    'niah_single_2':  {'gen_len': 64,  'metric': needle_score,      'metric_name': 'needle_score'},
    'niah_multikey_1': {'gen_len': 64,  'metric': needle_score,      'metric_name': 'needle_score'},
    'niah_multikey_2': {'gen_len': 64,  'metric': needle_score,      'metric_name': 'needle_score'},
    'niah_multivalue': {'gen_len': 128, 'metric': needle_score,      'metric_name': 'needle_score'},
    'niah_multiquery': {'gen_len': 128, 'metric': needle_score,      'metric_name': 'needle_score'},
    'vt':              {'gen_len': 30,  'metric': multi_words,       'metric_name': 'multi_words'},
    'fwe':             {'gen_len': 50,  'metric': multi_words,       'metric_name': 'multi_words'},
    'qa_1':            {'gen_len': 32,  'metric': string_match_part, 'metric_name': 'string_match_part'},
    'qa_2':            {'gen_len': 32,  'metric': string_match_part, 'metric_name': 'string_match_part'},
}


def load_ruler_data(data_dir, task_name):
    """Load RULER JSONL data."""
    jsonl_path = os.path.join(data_dir, task_name, 'validation.jsonl')
    examples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples


def evaluate_task(chat, tok, task_name, examples, config, num_samples=-1):
    """Evaluate a single RULER task."""
    task_cfg = TASK_CONFIG[task_name]
    gen_len = task_cfg['gen_len']
    if config.get("cot", False):
        gen_len *= 2

    gen_config = GenerationConfig(
        max_new_tokens=gen_len,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
    )

    if num_samples > 0:
        examples = examples[:num_samples]

    results = []
    total_score = 0.0

    for i, eg in enumerate(tqdm(examples, desc=f"Running {task_name}")):
        input_text = eg['input']
        ground_truth = eg['outputs']

        pred = chat(input_text, generation_config=gen_config, skip_special_tokens=True)

        # Compute score
        metric_fn = task_cfg['metric']
        if metric_fn in (multi_words, multi_number):
            score = metric_fn(pred, ground_truth)
        elif metric_fn == needle_score:
            # needle_score expects single string gt; try each and take max
            best = 0.0
            for gt in ground_truth:
                best = max(best, needle_score(pred, gt))
            score = best
        elif metric_fn == string_match_part:
            score = string_match_part(pred, ground_truth)
        else:
            score = 0.0

        total_score += score
        results.append({
            'index': eg.get('index', i),
            'prediction': pred,
            'ground_truth': ground_truth,
            'score': score,
        })

    avg_score = total_score / len(results) if results else 0.0
    return results, avg_score


def main():
    parser = argparse.ArgumentParser(description='RULER evaluation for OmniKV')
    parser.add_argument('--config_path', type=str, required=True, help='OmniKV config JSON path')
    parser.add_argument('--data_dir', type=str, required=True, help='RULER data directory (e.g., .../qwen3/102400)')
    parser.add_argument('--tasks', type=str, required=True, help='Comma-separated task names')
    parser.add_argument('--output_dir', type=str, default='results/ruler', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples per task (-1 for all)')
    args = parser.parse_args()

    task_names = [t.strip() for t in args.tasks.split(',')]

    # Load model
    config = read_config(args.config_path)
    chat, tok, max_len, o_dict = get_any_chat_api(args.config_path)

    if (eos := o_dict.get('eos_token_id', None)) is not None:
        tok.eos_token_id = eos

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each task
    summary = {}
    for task_name in task_names:
        if task_name not in TASK_CONFIG:
            print(f"[Warning] Unknown task: {task_name}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        examples = load_ruler_data(args.data_dir, task_name)
        results, avg_score = evaluate_task(chat, tok, task_name, examples, config, args.num_samples)

        # Save per-task results
        task_output = output_dir / f"preds_{task_name}.jsonl"
        with open(task_output, 'w') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        summary[task_name] = {
            'score': round(avg_score, 4),
            'num_samples': len(results),
            'metric': TASK_CONFIG[task_name]['metric_name'],
        }
        print(f"  Score: {avg_score:.4f} ({len(results)} samples)")

    # Print and save summary
    print(f"\n{'='*60}")
    print("RULER Summary")
    print(f"{'='*60}")
    print(f"{'Task':<25} {'Score':>8} {'Samples':>8}")
    print('-' * 45)
    total = 0.0
    count = 0
    for task_name, info in summary.items():
        print(f"{task_name:<25} {info['score']:>8.4f} {info['num_samples']:>8}")
        total += info['score']
        count += 1
    if count > 0:
        print('-' * 45)
        print(f"{'Average':<25} {total/count:>8.4f} {'' :>8}")

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
