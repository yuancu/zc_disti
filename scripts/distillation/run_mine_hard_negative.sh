#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BEDROCK_DIR="$PROJECT_DIR/data/bedrock"
RESULT_DIR="$PROJECT_DIR/artifacts/hard-negative"
NUM_NEGATIVES=20

# Dataset name -> bedrock output file, corpus file
declare -A BEDROCK_FILES=(
  ["cure_v1"]="$BEDROCK_DIR/cure_v1_batch_input.jsonl.out"
  ["hc3finance"]="$BEDROCK_DIR/hc3finance_batch_input.jsonl.out"
)

declare -A CORPUS_FILES=(
  ["cure_v1"]="/home/ubuntu/src/ir-fine-tune-evaluation/data/CUREv1_en/corpus.jsonl"
  ["hc3finance"]="/home/ubuntu/src/ir-fine-tune-evaluation/data/HC3Finance/corpus.jsonl"
)

# mine_hard_negative.py expects a specific directory layout:
#   output-dir/<dataset>/<subdir>/*_batch_input.jsonl.out
#   corpus-dir/<dataset>.jsonl
# Create a temp workspace with symlinks to satisfy this.
WORKSPACE=$(mktemp -d)
trap 'rm -rf "$WORKSPACE"' EXIT

OUTPUT_STAGING="$WORKSPACE/output"
CORPUS_STAGING="$WORKSPACE/corpus"
mkdir -p "$OUTPUT_STAGING" "$CORPUS_STAGING"

for dataset in "${!BEDROCK_FILES[@]}"; do
  mkdir -p "$OUTPUT_STAGING/$dataset/batch"
  ln -s "${BEDROCK_FILES[$dataset]}" "$OUTPUT_STAGING/$dataset/batch/"
  ln -s "${CORPUS_FILES[$dataset]}" "$CORPUS_STAGING/${dataset}.jsonl"
done

mkdir -p "$RESULT_DIR"

# Run each dataset separately
for dataset in "${!BEDROCK_FILES[@]}"; do
  echo "========================================"
  echo "Processing: $dataset"
  echo "========================================"
  python "$SCRIPT_DIR/mine_hard_negative.py" \
    --output-dir "$OUTPUT_STAGING" \
    --corpus-dir "$CORPUS_STAGING" \
    --result-dir "$RESULT_DIR" \
    --datasets "$dataset" \
    --num-negatives "$NUM_NEGATIVES"
  echo ""
done

echo "Done. Results saved to: $RESULT_DIR"
