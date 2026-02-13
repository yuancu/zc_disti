#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

BEDROCK_DIR="$PROJECT_DIR/data/bedrock"
EVAL_DIR="$PROJECT_DIR/data/eval"
RESULT_DIR="$PROJECT_DIR/artifacts/hard-negative-semantic"
NUM_NEGATIVES=20
MODEL_NAME="BAAI/bge-m3"
BATCH_SIZE=64

# Dataset name -> bedrock output file, corpus file
declare -A BEDROCK_FILES=(
  # ["cure_v1"]="$BEDROCK_DIR/cure_v1_batch_input.jsonl.out"
  # ["hc3finance"]="$BEDROCK_DIR/hc3finance_batch_input.jsonl.out"
  # ["aila-casedocs"]="$BEDROCK_DIR/aila-casedocs_batch_input.jsonl.out"
  # ["finqa"]="$BEDROCK_DIR/finqa_batch_input.jsonl.out"
  # ["financebench"]="$BEDROCK_DIR/financebench_batch_input.jsonl.out"
  # ["multi-cpr-video"]="$BEDROCK_DIR/multi-cpr-video_batch_input.jsonl.out"
  ["legal-summ"]="$BEDROCK_DIR/legal-summ_batch_input.jsonl.out"
  # ["legalquad"]="$BEDROCK_DIR/legalquad_batch_input.jsonl.out"
)

declare -A CORPUS_FILES=(
  # ["cure_v1"]="/home/ubuntu/src/ir-fine-tune-evaluation/data/CUREv1_en/corpus.jsonl"
  # ["hc3finance"]="$EVAL_DIR/HC3Finance/corpus.jsonl"
  # ["aila-casedocs"]="$EVAL_DIR/AILA_casedocs/corpus.jsonl"
  # ["finqa"]="$EVAL_DIR/finqa/corpus.jsonl"
  # ["financebench"]="$EVAL_DIR/financebench/corpus.jsonl"
  # ["multi-cpr-video"]="$EVAL_DIR/multi-cpr-video/corpus.jsonl"
  ["legal-summ"]="$EVAL_DIR/legal_summarization/corpus.jsonl"
  # ["legalquad"]="$EVAL_DIR/LegalQuAD/corpus.jsonl"
)

# mine_hard_negative_semantic.py expects the same directory layout as the BM25 version:
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
  python "$SCRIPT_DIR/mine_hard_negative_semantic.py" \
    --output-dir "$OUTPUT_STAGING" \
    --corpus-dir "$CORPUS_STAGING" \
    --result-dir "$RESULT_DIR" \
    --model-name "$MODEL_NAME" \
    --datasets "$dataset" \
    --num-negatives "$NUM_NEGATIVES" \
    --batch-size "$BATCH_SIZE"
  echo ""
done

echo "Done. Results saved to: $RESULT_DIR"
