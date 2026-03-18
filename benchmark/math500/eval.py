"""
MATH-500 evaluation script for OmniKV.
Computes both exact_match and math_verify accuracy,
broken down by level and problem type.

Usage:
    python -m benchmark.math500.eval --model my_model --cfg configs/example.json
    python -m benchmark.math500.eval --model my_model --cfg configs/qwen3_8b.json --verbose
"""
import os
import json
import argparse
import numpy as np

from benchmark.math500.metrics import (
    exact_match_score,
    math_verify_score,
    extract_answer_from_response,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="MATH-500 Evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True,
                        help="Path to model config JSON")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each problem's result")
    return parser.parse_args(args)


def evaluate(pred_path, verbose=False):
    """Evaluate predictions with both exact_match and math_verify."""
    predictions = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            predictions.append(json.loads(line))

    if not predictions:
        print("No predictions found!")
        return {}

    # Per-metric tracking
    em_scores = []
    mv_scores = []
    # Per-level
    em_by_level = {}
    mv_by_level = {}
    # Per-type
    em_by_type = {}
    mv_by_type = {}

    for i, item in enumerate(predictions):
        pred = item["pred"]
        gt_answer = item["answer"]

        em = exact_match_score(pred, gt_answer)
        mv = math_verify_score(pred, gt_answer)

        em_scores.append(em)
        mv_scores.append(mv)

        level = f"Level {item.get('level', '?')}"
        ptype = item.get("type", "unknown") or "unknown"

        em_by_level.setdefault(level, []).append(em)
        mv_by_level.setdefault(level, []).append(mv)
        em_by_type.setdefault(ptype, []).append(em)
        mv_by_type.setdefault(ptype, []).append(mv)

        if verbose:
            pred_answer = extract_answer_from_response(pred)
            em_s = "✓" if em > 0 else "✗"
            mv_s = "✓" if mv > 0 else "✗"
            print(f"[{i+1}] EM:{em_s} MV:{mv_s}  pred={pred_answer}  gt={gt_answer}  ({level}, {ptype})")

    def build_breakdown(scores_by_key):
        result = {}
        for key in sorted(scores_by_key.keys()):
            s = scores_by_key[key]
            result[key] = {
                "accuracy": round(100 * np.mean(s), 2),
                "correct": int(sum(s)),
                "total": len(s),
            }
        return result

    results = {
        "exact_match": {
            "overall_accuracy": round(100 * np.mean(em_scores), 2),
            "correct": int(sum(em_scores)),
            "total": len(em_scores),
            "by_level": build_breakdown(em_by_level),
            "by_type": build_breakdown(em_by_type),
        },
        "math_verify": {
            "overall_accuracy": round(100 * np.mean(mv_scores), 2),
            "correct": int(sum(mv_scores)),
            "total": len(mv_scores),
            "by_level": build_breakdown(mv_by_level),
            "by_type": build_breakdown(mv_by_type),
        },
    }

    return results


def print_results(results):
    for metric_name in ["exact_match", "math_verify"]:
        m = results[metric_name]
        print(f"\n{'='*55}")
        print(f" {metric_name}: {m['correct']}/{m['total']} = {m['overall_accuracy']}%")
        print(f"{'='*55}")

        print("  By Level:")
        for level, info in m["by_level"].items():
            print(f"    {level}: {info['correct']}/{info['total']} = {info['accuracy']}%")

        print("  By Type:")
        for ptype, info in m["by_type"].items():
            print(f"    {ptype}: {info['correct']}/{info['total']} = {info['accuracy']}%")


if __name__ == "__main__":
    args = parse_args()
    pred_dir = f"benchmark/math500/pred/{args.model}/{args.cfg}"
    pred_path = os.path.join(pred_dir, "math500.jsonl")

    if not os.path.exists(pred_path):
        print(f"Prediction file not found: {pred_path}")
        print("Run pred.py first to generate predictions.")
        exit(1)

    print(f"Evaluating: {pred_path}")
    results = evaluate(pred_path, verbose=args.verbose)

    print_results(results)

    # Save results
    result_path = os.path.join(pred_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\nResults saved to: {result_path}")
