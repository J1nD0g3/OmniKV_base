################################################################################
# RULER data generation for OmniKV
# Copied from ShadowKV's data generator (NVIDIA RULER, Apache-2.0)
#
# Usage: bash create_dataset.sh <MODEL_PATH> <MODEL_TEMPLATE_TYPE>
# Example: bash create_dataset.sh /path/to/Qwen3-8B-128k qwen3
################################################################################

SEQ_LENGTHS=(
    102400
)

MODEL_NAME=$1
MODEL_TEMPLATE_TYPE=$2

if [ -z "$MODEL_NAME" ] || [ -z "$MODEL_TEMPLATE_TYPE" ]; then
    echo "Usage: bash create_dataset.sh <MODEL_PATH> <MODEL_TEMPLATE_TYPE>"
    echo "Example: bash create_dataset.sh /path/to/Qwen3-8B-128k qwen3"
    exit 1
fi

echo "Model Name: $MODEL_NAME"
echo "Model Template Type: $MODEL_TEMPLATE_TYPE"

NUM_SAMPLES=96
REMOVE_NEWLINE_TAB=false
STOP_WORDS=""

if [ -z "${STOP_WORDS}" ]; then
    STOP_WORDS=""
else
    STOP_WORDS="--stop_words \"${STOP_WORDS}\""
fi

if [ "${REMOVE_NEWLINE_TAB}" = false ]; then
    REMOVE_NEWLINE_TAB=""
else
    REMOVE_NEWLINE_TAB="--remove_newline_tab"
fi

synthetic=(
    "niah_single_1"
    "niah_single_2"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "fwe"
    "qa_1"
    "qa_2"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do

    RESULTS_DIR="${SCRIPT_DIR}/data/${MODEL_TEMPLATE_TYPE}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/"
    mkdir -p ${DATA_DIR}

    for TASK in "${synthetic[@]}"; do
        echo "TASK: ${TASK}, MAX_SEQ_LENGTH: ${MAX_SEQ_LENGTH}"
        python "${SCRIPT_DIR}/prepare.py" \
            --save_dir ${DATA_DIR} \
            --task ${TASK} \
            --tokenizer_path ${MODEL_NAME} \
            --tokenizer_type hf \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            ${REMOVE_NEWLINE_TAB}
    done

done

echo "Done. Data saved to ${SCRIPT_DIR}/data/${MODEL_TEMPLATE_TYPE}/"
