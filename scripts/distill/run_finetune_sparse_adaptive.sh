#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_finetune_sparse_adaptive.sh <ratio>
# Example: bash run_finetune_sparse_adaptive.sh 0.1

RATIO="${1:?Usage: $0 <ratio>}"

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate pytorch 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_DIR="$PROJECT_DIR/artifacts/teacher-score-gemma2-seeded"
OUTPUT_DIR="$PROJECT_DIR/artifacts/models/distilled-sparse-gte-adaptive-r${RATIO}"
MODEL_NAME="opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"
NUM_GPUS=4

DATASETS=("cure_v1" "aila-casedocs" "financebench" "finqa" "hc3finance" "legalquad" "legal-summ")

mkdir -p "$OUTPUT_DIR"

for dataset in "${DATASETS[@]}"; do
  if [ -d "$OUTPUT_DIR/$dataset" ] && [ -f "$OUTPUT_DIR/$dataset/modules.json" ]; then
    echo "Skipping $dataset (already trained)"
    continue
  fi
  rm -rf "$OUTPUT_DIR/$dataset"
  echo "========================================"
  echo "Training: $dataset (adaptive, ratio=${RATIO})"
  echo "========================================"
  torchrun --nproc_per_node="$NUM_GPUS" "$SCRIPT_DIR/finetune_sparse.py" \
    --dataset "$dataset" \
    --data_path "$INPUT_DIR/${dataset}.json" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR/${dataset}" \
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_steps 500 \
    --normalize_scores \
    --teacher_score_scale_factor 20.0 \
    --adaptive_regularizer \
    --document_regularizer_weight "$RATIO" \
    --skip_eval
  echo ""
done

echo "Done. All models trained with adaptive ratio=${RATIO}."
