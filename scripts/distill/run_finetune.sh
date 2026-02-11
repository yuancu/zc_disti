#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_DIR="$PROJECT_DIR/artifacts/teacher-score-gemma2"
OUTPUT_DIR="$PROJECT_DIR/artifacts/distilled"

MODEL_NAME="BAAI/bge-m3"
# Derive a short name from the model for the output directory
BASE_MODEL_NAME=$(echo "$MODEL_NAME" | sed 's|.*/||')
NUM_GPUS=4

# DATASETS=("hc3finance" "cure_v1")
DATASETS=("aila-casedocs" "legal-summ" "legalquad" "finqa" "financebench")

mkdir -p "$OUTPUT_DIR"

for dataset in "${DATASETS[@]}"; do
  echo "========================================"
  echo "Finetuning: $dataset (model: $MODEL_NAME)"
  echo "========================================"
  torchrun --nproc_per_node="$NUM_GPUS" "$SCRIPT_DIR/finetune.py" \
    --dataset "$dataset" \
    --data_path "$INPUT_DIR/${dataset}.json" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR/${BASE_MODEL_NAME}_${dataset}" \
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_steps 500 \
    --skip_eval
  echo ""
done

echo "Done. Models saved to: $OUTPUT_DIR"
