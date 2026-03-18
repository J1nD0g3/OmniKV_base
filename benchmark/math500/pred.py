"""
MATH-500 prediction script for OmniKV.
Follows the same pattern as benchmark/long_bench/pred.py.

Usage:
    python -m benchmark.math500.pred --model my_model --cfg configs/example.json
    python -m benchmark.math500.pred --model my_model --cfg configs/qwen3_8b.json --max_problems 50
"""
import os
import json
import time
import argparse
from tqdm import tqdm
import numpy as np
import random

import torch
from transformers import AutoTokenizer

from infer import get_any_chat_api
from tiny_tools.read_json import read_config


MATH_PROMPT = """Solve the following math problem step by step. Put your final answer within \\boxed{{}}.

Problem: {problem}"""


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="MATH-500 Prediction")
    parser.add_argument("--model", type=str, default="my_model",
                        choices=["my_model"])
    parser.add_argument("--cfg", type=str, required=True,
                        help="Path to model config JSON")
    parser.add_argument("--max_problems", type=int, default=None,
                        help="Limit number of problems (for debugging)")
    parser.add_argument("--max_gen", type=int, default=2048,
                        help="Max generation tokens")
    parser.add_argument("--data_path", type=str,
                        default="benchmark/math500/data/math500.jsonl",
                        help="Path to MATH-500 JSONL data")
    return parser.parse_args(args)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_dataset(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_pred(data, model, tokenizer, max_length, args):
    """Generate predictions for all math problems."""
    out_dir = f"benchmark/math500/pred/{args.model}/{args.cfg}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "math500.jsonl")

    # Clear output file
    with open(out_path, "w") as _:
        pass

    for json_obj in tqdm(data, desc="MATH-500"):
        prompt = MATH_PROMPT.format(problem=json_obj["problem"])

        # Truncate if needed (math problems are usually short, but just in case)
        tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized) > max_length:
            half = int(max_length / 2)
            prompt = (tokenizer.decode(tokenized[:half], skip_special_tokens=True) +
                      tokenizer.decode(tokenized[-half:], skip_special_tokens=True))

        output = model(
            prompt,
            generation_config=None,
            max_new_tokens=args.max_gen,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            skip_special_tokens=True,
        )

        record = {
            "pred": output,
            "answer": json_obj["answer"],
            "problem": json_obj["problem"],
            "level": json_obj.get("level", ""),
            "type": json_obj.get("type", ""),
        }
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    return out_path


if __name__ == "__main__":
    args = parse_args()
    seed_everything(42)

    print(f"Loading model from config: {args.cfg}")
    model, tokenizer, max_length, other_kwargs = get_any_chat_api(args.cfg)
    tokenizer.eos_token_id = other_kwargs.get("eos_token_id", tokenizer.eos_token_id)
    print(f"Model loaded. max_context_len={max_length}")

    print(f"Loading dataset from: {args.data_path}")
    data = load_dataset(args.data_path)
    if args.max_problems is not None:
        data = data[:args.max_problems]
    print(f"Loaded {len(data)} problems")

    torch.cuda.empty_cache()
    out_path = get_pred(data, model, tokenizer, max_length, args)
    print(f"Predictions saved to: {out_path}")
    print("Run eval.py to compute accuracy.")
