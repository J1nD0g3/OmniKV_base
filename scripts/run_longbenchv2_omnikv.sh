#!/bin/bash
# LongBench-v2 evaluation with OmniKV
#
# Usage:
#   bash run_longbenchv2_omnikv.sh CONFIG_PATH [NUM_SAMPLES]
#
# Examples:
#   bash run_longbenchv2_omnikv.sh configs/qwen3_8b_32k.json
#   bash run_longbenchv2_omnikv.sh configs/qwen3_8b_32k.json 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OMNIKV_DIR="$(dirname "$SCRIPT_DIR")"

# ===================== Config =====================
CONFIG_PATH="${1:?Usage: $0 CONFIG_PATH [NUM_SAMPLES]}"
NUM_SAMPLES="${2:-0}"
# ==================================================

# Activate conda (works with anaconda, miniconda, miniforge)
CONDA_SH="${CONDA_PREFIX:-$HOME/anaconda3}/etc/profile.d/conda.sh"
[ -f "$CONDA_SH" ] || CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate omnikv

cd "${OMNIKV_DIR}"

echo "=== LongBench-v2 OmniKV ==="
echo "Config: ${CONFIG_PATH}"
echo "Samples: ${NUM_SAMPLES} (0=all)"

EXTRA_ARGS=""
if [ "$NUM_SAMPLES" -gt 0 ]; then
    EXTRA_ARGS="--n $NUM_SAMPLES"
fi

python benchmark/long_bench_v2/pred.py \
    --cfg "${CONFIG_PATH}" \
    ${EXTRA_ARGS} \
    2>&1

echo "Done."
