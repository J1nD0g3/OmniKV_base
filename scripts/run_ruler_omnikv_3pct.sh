#!/bin/bash
# RULER OmniKV 3% test (3 samples per task)
# Quick estimation run before full evaluation

# ===================== Config =====================
MODEL_PATH="/workspace/models/Qwen3-8B-128k"
MODEL_TEMPLATE="qwen3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OMNIKV_DIR="$(dirname "$SCRIPT_DIR")"
RULER_DATA_DIR="${OMNIKV_DIR}/data/ruler/data/${MODEL_TEMPLATE}/102400"
RULER_TASKS="niah_single_1,niah_single_2,niah_multikey_1,niah_multikey_2,niah_multivalue,niah_multiquery,vt,fwe,qa_1,qa_2"
CONFIG_PATH="configs/qwen3_8b_100k_ruler.json"
NUM_SAMPLES=3
# ==================================================

cd "${OMNIKV_DIR}"

# Generate RULER data if not exists
TASK_LIST=(niah_single_1 niah_single_2 niah_multikey_1 niah_multikey_2 niah_multivalue niah_multiquery vt fwe qa_1 qa_2)
MISSING=0
for t in "${TASK_LIST[@]}"; do
    if [ ! -f "${RULER_DATA_DIR}/${t}/validation.jsonl" ]; then
        echo "[INFO] Missing RULER data: ${t}"
        MISSING=1
    fi
done
if [ "$MISSING" -eq 1 ]; then
    echo "[INFO] Generating RULER data..."
    cd "${OMNIKV_DIR}/data/ruler"
    bash create_dataset.sh "$MODEL_PATH" "$MODEL_TEMPLATE"
    cd "${OMNIKV_DIR}"
    echo "[INFO] RULER data generation complete."
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${OMNIKV_DIR}/logs/ruler_3pct_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=== RULER OmniKV 3% test ===" | tee "$LOG_DIR/run.log"
echo "Start: $(date)" | tee -a "$LOG_DIR/run.log"
echo "Model: $MODEL_PATH" | tee -a "$LOG_DIR/run.log"
echo "Samples per task: $NUM_SAMPLES" | tee -a "$LOG_DIR/run.log"
echo "Log dir: $LOG_DIR" | tee -a "$LOG_DIR/run.log"

PYTHONPATH=./ python benchmark/ruler/eval_ruler.py \
    --config_path "$CONFIG_PATH" \
    --data_dir "$RULER_DATA_DIR" \
    --tasks "$RULER_TASKS" \
    --output_dir "$LOG_DIR" \
    --num_samples $NUM_SAMPLES \
    2>&1 | tee -a "$LOG_DIR/run.log"

echo "End: $(date)" | tee -a "$LOG_DIR/run.log"
echo "Results saved to: $LOG_DIR"
