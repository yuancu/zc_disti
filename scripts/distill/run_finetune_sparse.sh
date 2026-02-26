#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_DIR="$PROJECT_DIR/artifacts/teacher-score-gemma2-seeded"
OUTPUT_DIR="$PROJECT_DIR/artifacts/models/distilled-sparse-multilingual-500steps-seeded"

# MODEL_NAME="opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"
MODEL_NAME="opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
NUM_GPUS=3

# DATASETS=("multi-cpr-video")
DATASETS=("multi-cpr-ecom" "multi-cpr-medical" "cure_v1" "aila-casedocs" "financebench" "finqa" "hc3finance" "legalquad" "legal-summ")

mkdir -p "$OUTPUT_DIR"

for dataset in "${DATASETS[@]}"; do
  echo "========================================"
  echo "Sparse finetuning: $dataset (model: $MODEL_NAME)"
  echo "========================================"
  torchrun --nproc_per_node="$NUM_GPUS" "$SCRIPT_DIR/finetune_sparse.py" \
    --dataset "$dataset" \
    --data_path "$INPUT_DIR/${dataset}.json" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR/${dataset}" \
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_steps 500 \
    --skip_eval
  echo ""
done

echo "Done. Models saved to: $OUTPUT_DIR"
