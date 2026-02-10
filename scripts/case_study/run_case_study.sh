#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Case Study: Compare fine-tuned vs baseline embedding models
# ============================================================

# --- Common settings ---
BASELINE_MODEL="BAAI/bge-m3"
NUM_GPUS=4
BATCH_SIZE=64
SKIP_EMBEDDING=false  # Set to true to reuse cached embeddings
NUM_SAMPLES=5
SEED=42

# --- Dataset configurations ---
# Each entry: EVAL_DATA_DIR FINETUNED_MODEL OUTPUT_DIR
DATASETS=(
    "/home/ubuntu/src/ir-fine-tune-evaluation/data/HC3Finance artifacts/distilled-500steps/bge-m3_hc3finance artifacts/case_study/hc3finance"
    "/home/ubuntu/src/ir-fine-tune-evaluation/data/CUREv1_en artifacts/distilled-500steps/bge-m3_cure_v1 artifacts/case_study/cure_v1"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SKIP_FLAG=""
if [ "${SKIP_EMBEDDING}" = true ]; then
    SKIP_FLAG="--skip-embedding"
fi

for entry in "${DATASETS[@]}"; do
    read -r EVAL_DATA_DIR FINETUNED_MODEL OUTPUT_DIR <<< "${entry}"

    # Resolve relative paths against the project root
    [[ "${FINETUNED_MODEL}" != /* ]] && FINETUNED_MODEL="${PROJECT_DIR}/${FINETUNED_MODEL}"
    [[ "${OUTPUT_DIR}" != /* ]] && OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"

    DATASET_NAME="$(basename "${EVAL_DATA_DIR}")"

    echo ""
    echo "============================================================"
    echo "  Case study: ${DATASET_NAME}"
    echo "============================================================"
    echo "  Data:      ${EVAL_DATA_DIR}"
    echo "  Finetuned: ${FINETUNED_MODEL}"
    echo "  Output:    ${OUTPUT_DIR}"
    echo "============================================================"

    # Step 1: Run main case study (embeddings + retrieval + BM25)
    python "${SCRIPT_DIR}/case_study.py" \
        --eval-data-dir "${EVAL_DATA_DIR}" \
        --finetuned-model "${FINETUNED_MODEL}" \
        --baseline-model "${BASELINE_MODEL}" \
        --output-dir "${OUTPUT_DIR}" \
        --num-gpus "${NUM_GPUS}" \
        --batch-size "${BATCH_SIZE}" \
        ${SKIP_FLAG}

    # Step 2: Sample typical cases for qualitative analysis
    python "${SCRIPT_DIR}/sample_typical_cases.py" \
        --output-dir "${OUTPUT_DIR}" \
        --num-samples "${NUM_SAMPLES}" \
        --seed "${SEED}"

    echo ""
    echo "Done: ${DATASET_NAME} -> ${OUTPUT_DIR}"
    echo ""
done

echo "========================================"
echo "All case studies complete."
echo "========================================"
