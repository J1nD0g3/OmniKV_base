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
    'niah_single_1':  {'gen_len': 64,  'metric': needle_score,      'metric_name': 'needle_score',      'category': 'Retrieval'},
    'niah_single_2':  {'gen_len': 64,  'metric': needle_score,      'metric_name': 'needle_score',      'category': 'Retrieval'},
    'niah_multikey_1': {'gen_len': 64,  'metric': needle_score,      'metric_name': 'needle_score',      'category': 'Multi-key Retrieval'},
    'niah_multikey_2': {'gen_len': 64,  'metric': needle_score,      'metric_name': 'needle_score',      'category': 'Multi-key Retrieval'},
    'niah_multivalue': {'gen_len': 128, 'metric': needle_score,      'metric_name': 'needle_score',      'category': 'Multi-value Retrieval'},
    'niah_multiquery': {'gen_len': 128, 'metric': needle_score,      'metric_name': 'needle_score',      'category': 'Multi-query Retrieval'},
    'vt':              {'gen_len': 30,  'metric': multi_words,       'metric_name': 'multi_words',       'category': 'Variable Tracking'},
    'fwe':             {'gen_len': 50,  'metric': multi_words,       'metric_name': 'multi_words',       'category': 'Aggregation'},
    'qa_1':            {'gen_len': 32,  'metric': string_match_part, 'metric_name': 'string_match_part', 'category': 'Question Answering'},
    'qa_2':            {'gen_len': 32,  'metric': string_match_part, 'metric_name': 'string_match_part', 'category': 'Question Answering'},
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

    pbar = tqdm(examples, desc=f"{task_name} avg=-.----")
    for i, eg in enumerate(pbar):
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

        running_avg = total_score / len(results)
        pbar.set_description(f"{task_name} avg={running_avg:.4f}")

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

    # ---- summary.txt: per-task + per-category + model info ----
    from datetime import datetime

    # Compute KV cache ratio (same logic as infer.py)
    # Read num_hidden_layers from model config.json since it's not in the eval config
    model_name = config.get("model_name", "")
    model_config_path = os.path.join(model_name, "config.json")
    num_layers = 0
    if os.path.exists(model_config_path):
        with open(model_config_path) as _f:
            num_layers = json.load(_f).get("num_hidden_layers", 0)
    do_sel_layers = [int(i) for i in config.get("do_select_layers", "").split(",")]
    num_wait = config.get("num_wait_load_layers", 2)
    token_ratio = config.get("num_of_selected_tokens", 4096)
    full_attn_layers = list(range(0, do_sel_layers[0])) + do_sel_layers + [num_layers]
    num_full, num_sparse = 0, 0
    for _l in full_attn_layers:
        _r = num_layers
        for _i in range(_l + 1, num_layers + 1):
            if _i in full_attn_layers:
                _r = _i
                break
        num_full += min(_r, _l + num_wait + 1) - _l
        num_sparse += max(0, _r - min(_r, _l + num_wait + 1))
    if isinstance(token_ratio, float):
        overall_kv_ratio = (num_full * 1.0 + num_sparse * token_ratio) / num_layers
    else:
        overall_kv_ratio = None

    # Build category aggregation
    cat_scores = {}  # category -> list of (task, score)
    for task_name, info in summary.items():
        cat = TASK_CONFIG.get(task_name, {}).get('category', 'Other')
        cat_scores.setdefault(cat, []).append((task_name, info['score']))

    lines = []
    lines.append("=" * 65)
    lines.append("RULER Benchmark Results")
    lines.append("=" * 65)
    lines.append("")

    # Experiment info
    lines.append("[Experiment]")
    lines.append(f"  Date           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Config         : {args.config_path}")
    lines.append(f"  Data           : {args.data_dir}")
    lines.append(f"  Samples/task   : {args.num_samples if args.num_samples > 0 else 'all (96)'}")
    lines.append("")

    # Model info
    lines.append("[Model]")
    lines.append(f"  model_name     : {config.get('model_name', 'N/A')}")
    lines.append(f"  model_arch     : {config.get('model_arch', 'N/A')}")
    lines.append(f"  num_layers     : {num_layers}")
    lines.append(f"  max_context_len: {config.get('max_context_len', 'N/A')}")
    lines.append(f"  use_flash_attn : {config.get('use_flash_attn', False)}")
    lines.append(f"  enable_thinking: {config.get('enable_thinking', False)}")
    lines.append("")

    # KV cache info
    lines.append("[KV Cache]")
    lines.append(f"  model_cls            : {config.get('model_cls', 'N/A')}")
    lines.append(f"  cache_cls            : {config.get('cache_cls', 'N/A')}")
    lines.append(f"  do_select_layers     : {config.get('do_select_layers', 'N/A')}")
    lines.append(f"  num_wait_load_layers : {num_wait}")
    lines.append(f"  selector_cls         : {config.get('selector_cls', 'N/A')}")
    lines.append(f"  window_size          : {config.get('window_size', 'N/A')}")
    lines.append(f"  dense_more           : {config.get('dense_more', 'N/A')}")
    if isinstance(token_ratio, float):
        lines.append(f"  token_select_ratio   : {token_ratio:.4f} ({token_ratio*100:.2f}%)")
    else:
        lines.append(f"  num_selected_tokens  : {token_ratio} (absolute)")
    lines.append(f"  full_cache_layers    : {num_full}/{num_layers}")
    lines.append(f"  sparse_cache_layers  : {num_sparse}/{num_layers}")
    if overall_kv_ratio is not None:
        lines.append(f"  overall_kv_ratio     : {overall_kv_ratio:.4f} ({overall_kv_ratio*100:.2f}%)")
    lines.append("")

    # Per-task table
    lines.append("=" * 65)
    lines.append("Per-Task Results")
    lines.append("=" * 65)
    lines.append(f"{'Task':<25} {'Category':<25} {'Score':>8} {'N':>5}")
    lines.append("-" * 65)
    for task_name, info in summary.items():
        cat = TASK_CONFIG.get(task_name, {}).get('category', 'Other')
        lines.append(f"{task_name:<25} {cat:<25} {info['score']:>8.4f} {info['num_samples']:>5}")
    lines.append("-" * 65)
    if count > 0:
        lines.append(f"{'OVERALL AVERAGE':<25} {'':<25} {total/count:>8.4f}")
    lines.append("")

    # Per-category table
    lines.append("=" * 65)
    lines.append("Per-Category Results")
    lines.append("=" * 65)
    lines.append(f"{'Category':<30} {'Avg Score':>10} {'Tasks':>6}")
    lines.append("-" * 50)
    cat_overall = 0.0
    cat_count = 0
    for cat in sorted(cat_scores.keys()):
        tasks = cat_scores[cat]
        cat_avg = sum(s for _, s in tasks) / len(tasks)
        lines.append(f"{cat:<30} {cat_avg:>10.4f} {len(tasks):>6}")
        cat_overall += cat_avg
        cat_count += 1
    if cat_count > 0:
        lines.append("-" * 50)
        lines.append(f"{'CATEGORY AVERAGE':<30} {cat_overall/cat_count:>10.4f}")
    lines.append("")

    summary_txt = "\n".join(lines)
    print(summary_txt)

    txt_path = output_dir / 'summary.txt'
    with open(txt_path, 'w') as f:
        f.write(summary_txt + "\n")

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
