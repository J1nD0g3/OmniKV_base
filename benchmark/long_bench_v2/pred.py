"""LongBench-v2 evaluation for OmniKV.

Loads THUDM/LongBench-v2 (503 multiple-choice samples), runs inference via
the same get_any_chat_api() used by long_bench/pred.py, and scores with
exact-match accuracy on A/B/C/D answers.
"""

import os
import sys
import json
import time
import datetime
import argparse

import numpy as np
import random
import torch
from tqdm import tqdm
from datasets import load_dataset as hf_load_dataset

# Project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'benchmark', 'long_bench'))

from tiny_tools.read_json import read_config
from infer import get_any_chat_api
import infer as infer_module
from metrics import longbenchv2_extract_answer, longbenchv2_accuracy


# 0-shot CoT prompt (matches ShadowKV & DiffKV templates)
PROMPT_TEMPLATE = (
    "Please read the following text and answer the questions below.\n\n"
    "<text>\n{context}\n</text>\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n"
    "(A) {choice_A}\n"
    "(B) {choice_B}\n"
    "(C) {choice_C}\n"
    "(D) {choice_D}\n\n"
    "Let's think step by step:"
)

MAX_GEN = 1024  # CoT reasoning budget


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def build_chat(tokenizer, prompt, model_name, enable_thinking=False):
    """Apply chat template (same logic as long_bench/pred.py)."""
    if "llama-2" in model_name.lower():
        prompt = f"[INST]{prompt}[/INST]"
    elif "qwen3" in model_name.lower():
        if enable_thinking:
            prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        else:
            prompt = f"<|im_start|>user\n{prompt}\n/no_think<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True, help="Model config yaml path")
    p.add_argument("--n", type=int, default=0, help="Number of samples (0 for all)")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    seed_everything(42)

    d_cfg = read_config(args.cfg)
    enable_thinking = d_cfg.get("enable_thinking", False)
    max_context_len = d_cfg.get("max_context_len", 32000)
    model_name = d_cfg.get("model_name", "unknown")
    model_short = os.path.basename(model_name)

    # Load model
    model, tokenizer, max_length, other_kwargs = get_any_chat_api(args.cfg)
    tokenizer.eos_token_id = other_kwargs['eos_token_id']
    if max_length is None:
        max_length = max_context_len

    # Load dataset
    dataset = hf_load_dataset('THUDM/LongBench-v2', split='train')
    if args.n > 0:
        dataset = dataset.select(range(min(args.n, len(dataset))))
    print(f"[Info] LongBench-v2: {len(dataset)} samples, max_length={max_length}")

    # Setup logging
    think_str = "think-on" if enable_thinking else "think-off"
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(ROOT_DIR, "logs", f"{model_short}_longbenchv2_{think_str}_{timestamp}")
    samples_dir = os.path.join(run_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    pred_path = os.path.join(run_dir, "predictions.jsonl")

    exp_start = time.time()
    scores = []
    sample_details = []

    for i in tqdm(range(len(dataset)), desc="LongBench-v2"):
        sample = dataset[i]

        prompt = PROMPT_TEMPLATE.format(
            context=sample['context'],
            question=sample['question'],
            choice_A=sample['choice_A'],
            choice_B=sample['choice_B'],
            choice_C=sample['choice_C'],
            choice_D=sample['choice_D'],
        )

        # Middle truncation
        tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized) > max_length:
            half = max_length // 2
            prompt = (tokenizer.decode(tokenized[:half], skip_special_tokens=True) +
                      tokenizer.decode(tokenized[-half:], skip_special_tokens=True))

        # Apply chat template
        prompt = build_chat(tokenizer, prompt, model_name, enable_thinking)

        # Inference
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt").to(0)
        context_length = input_ids.input_ids.shape[-1]

        _max_gen = MAX_GEN
        if enable_thinking:
            _max_gen = max(MAX_GEN, max_context_len - context_length)

        output = model(
            prompt, generation_config=None,
            max_new_tokens=_max_gen,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            skip_special_tokens=True,
        )
        if not isinstance(output, str):
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        else:
            pred = output

        # Score
        score = longbenchv2_accuracy(pred, sample['answer'])
        scores.append(score)

        meta = infer_module.last_inference_meta.copy()
        detail = {
            "index": i,
            "_id": sample['_id'],
            "domain": sample['domain'],
            "difficulty": sample['difficulty'],
            "answer": sample['answer'],
            "pred_answer": longbenchv2_extract_answer(pred),
            "score": score,
            "input_len": meta.get("input_len", context_length),
            "output_len": meta.get("output_len"),
            "num_selected_kv": meta.get("num_selected_kv"),
        }
        sample_details.append(detail)

        with open(pred_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, **detail}, f, ensure_ascii=False)
            f.write('\n')

    exp_elapsed = time.time() - exp_start
    avg_score = np.mean(scores) * 100

    # Breakdown by domain and difficulty
    domain_scores = {}
    diff_scores = {}
    for d in sample_details:
        domain_scores.setdefault(d['domain'], []).append(d['score'])
        diff_scores.setdefault(d['difficulty'], []).append(d['score'])

    # Build summary
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0

    lines = []
    lines.append("=" * 70)
    lines.append("OmniKV LongBench-v2 Experiment Log")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'Timestamp:':<20s} {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"{'Config:':<20s} {args.cfg}")
    lines.append(f"{'Model:':<20s} {model_name}")
    lines.append(f"{'Thinking mode:':<20s} {'ON' if enable_thinking else 'OFF'}")
    lines.append(f"{'Max context len:':<20s} {max_length}")
    lines.append(f"{'Samples:':<20s} {len(dataset)}")
    lines.append("")
    lines.append("--- Hardware ---")
    lines.append(f"{'GPU:':<20s} {gpu_name}")
    lines.append(f"{'Peak GPU memory:':<20s} {peak_mem:.2f} GB")
    lines.append(f"{'Total elapsed:':<20s} {exp_elapsed:.1f}s ({exp_elapsed/60:.1f}min)")
    lines.append("")
    lines.append("--- Results ---")
    lines.append(f"{'Overall accuracy:':<20s} {avg_score:.2f}%")
    lines.append("")
    lines.append("By domain:")
    for domain in sorted(domain_scores.keys()):
        s = domain_scores[domain]
        lines.append(f"  {domain:<45s} {np.mean(s)*100:5.2f}% ({sum(s):.0f}/{len(s)})")
    lines.append("")
    lines.append("By difficulty:")
    for diff in sorted(diff_scores.keys()):
        s = diff_scores[diff]
        lines.append(f"  {diff:<15s} {np.mean(s)*100:5.2f}% ({sum(s):.0f}/{len(s)})")
    lines.append("")

    # Inference stats
    input_lens = [d['input_len'] for d in sample_details if d.get('input_len')]
    output_lens = [d['output_len'] for d in sample_details if d.get('output_len')]
    selected_kvs = [d['num_selected_kv'] for d in sample_details if d.get('num_selected_kv')]
    lines.append("--- Inference Stats ---")
    if input_lens:
        lines.append(f"  input_len:  avg={np.mean(input_lens):.0f}  [{min(input_lens)}-{max(input_lens)}]")
    if output_lens:
        lines.append(f"  output_len: avg={np.mean(output_lens):.0f}  [{min(output_lens)}-{max(output_lens)}]")
    if selected_kvs and input_lens:
        ratios = [s / i for s, i in zip(selected_kvs, input_lens) if i > 0]
        lines.append(f"  selected_kv: avg={np.mean(selected_kvs):.0f}  ratio={np.mean(ratios):.4f}")
    lines.append("")
    lines.append("=" * 70)

    summary_text = "\n".join(lines)

    # Save
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(summary_text)

    summary_json = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": args.cfg,
        "model": model_name,
        "enable_thinking": enable_thinking,
        "num_samples": len(dataset),
        "accuracy": round(avg_score, 2),
        "domain_scores": {k: round(np.mean(v) * 100, 2) for k, v in domain_scores.items()},
        "difficulty_scores": {k: round(np.mean(v) * 100, 2) for k, v in diff_scores.items()},
        "total_elapsed_s": round(exp_elapsed, 2),
        "peak_gpu_memory_gb": round(peak_mem, 2),
        "gpu": gpu_name,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary_json, f, indent=4)

    with open(os.path.join(samples_dir, "longbenchv2.json"), "w") as f:
        json.dump(sample_details, f, indent=2, ensure_ascii=False)

    print(f"\n{summary_text}")
    print(f"\nLogs saved to: {run_dir}/")
