#!/usr/bin/env bash
set -euo pipefail

# Fine-tune sparse models using MarginMSE distillation (paper setting)
# Based on: "From Distillation to Hard Negative Sampling" (Formal et al., 2022)

# Ensure correct conda env (transformers 4.45.2, not 5.0.0)
TORCHRUN=/opt/conda/envs/pytorch/bin/torchrun

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_DIR="$PROJECT_DIR/artifacts/teacher-score-gemma2-seeded"
OUTPUT_DIR="$PROJECT_DIR/artifacts/models/distilled-sparse-marginmse"

MODEL_NAME="opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"
NUM_GPUS=4

DATASETS=("cure_v1" "aila-casedocs" "esci" "financebench" "finqa" "hc3finance" "legalquad" "legal-summ")

mkdir -p "$OUTPUT_DIR"

for dataset in "${DATASETS[@]}"; do
  DATA_FILE="$INPUT_DIR/${dataset}.json"
  if [[ ! -f "$DATA_FILE" ]]; then
    echo "WARNING: $DATA_FILE not found, skipping $dataset"
    continue
  fi

  echo "========================================"
  echo "MarginMSE finetuning: $dataset (model: $MODEL_NAME)"
  echo "========================================"
  "$TORCHRUN" --nproc_per_node="$NUM_GPUS" "$SCRIPT_DIR/finetune_sparse_marginmse.py" \
    --dataset "$dataset" \
    --data_path "$DATA_FILE" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR/${dataset}" \
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_steps 500 \
    --skip_eval
  echo ""
done

echo "Done. Models saved to: $OUTPUT_DIR"
