#!/bin/bash
# InfiniteBench evaluation with Qwen3-8B using OmniKV
# 128k context, 10 tasks (code_run/math_calc excluded)
# token_ratio=0.067, overall KV ratio ~27.4%

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OMNIKV_DIR="$(dirname "$SCRIPT_DIR")"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate omnikv
cd "${OMNIKV_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${OMNIKV_DIR}/logs/Qwen3-8B_infinitebench_omnikv_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=== InfiniteBench OmniKV ===" | tee "$LOG_DIR/run.log"
echo "Start: $(date)" | tee -a "$LOG_DIR/run.log"
echo "Model: Qwen/Qwen3-8B" | tee -a "$LOG_DIR/run.log"
echo "Dataset: InfiniteBench (10 tasks)" | tee -a "$LOG_DIR/run.log"
echo "Max context: 128000" | tee -a "$LOG_DIR/run.log"
echo "token_ratio: 0.067" | tee -a "$LOG_DIR/run.log"
echo "Log dir: $LOG_DIR" | tee -a "$LOG_DIR/run.log"

PYTHONPATH=./ bash shells/eval/eval_any_inf.sh configs/qwen3_8b_128k.json \
    2>&1 | tee -a "$LOG_DIR/run.log"

echo "End: $(date)" | tee -a "$LOG_DIR/run.log"
echo "Results saved to: $LOG_DIR"
