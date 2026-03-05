#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate pytorch 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_DIR="$PROJECT_DIR/artifacts/teacher-score-gemma2-seeded"
OUTPUT_DIR="$PROJECT_DIR/artifacts/models/distilled-sparse-v2distill"

MODEL_NAME="opensearch-project/opensearch-neural-sparse-encoding-v2-distill"
NUM_GPUS=3

DATASETS=("aila-casedocs" "cure_v1" "financebench" "finqa" "hc3finance" "legal-summ" "legalquad")

mkdir -p "$OUTPUT_DIR"

for dataset in "${DATASETS[@]}"; do
  echo "========================================"
  echo "Sparse finetuning: $dataset (model: $MODEL_NAME)"
  echo "========================================"
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node="$NUM_GPUS" "$SCRIPT_DIR/finetune_sparse.py" \
    --dataset "$dataset" \
    --data_path "$INPUT_DIR/${dataset}.json" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR/${dataset}" \
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_steps 500 \
    --normalize_scores \
    --teacher_score_scale_factor 30.0 \
    --document_regularizer_weight 0.002 \
    --query_regularizer_weight 0.002 \
    --symmetric \
    --skip_eval
  echo ""
done

echo "Done. Models saved to: $OUTPUT_DIR"
