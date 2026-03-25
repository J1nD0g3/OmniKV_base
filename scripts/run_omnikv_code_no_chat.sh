#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OMNIKV_DIR="$(dirname "$SCRIPT_DIR")"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate omnikv
export PYTHONPATH="${OMNIKV_DIR}"
cd "${OMNIKV_DIR}"

echo "=== OmniKV Code Completion (no chat template) ==="

python benchmark/long_bench/pred.py \
    --model my_model \
    --cfg configs/qwen3_8b_no_chat.json \
    --task lcc,repobench-p \
    2>&1 | tee "${OMNIKV_DIR}/logs/code_no_chat_pred.log"

python benchmark/long_bench/eval.py \
    --model my_model \
    --cfg configs/qwen3_8b_no_chat.json \
    --task lcc,repobench-p \
    2>&1 | tee "${OMNIKV_DIR}/logs/code_no_chat_eval.log"

echo "=== Done ==="
