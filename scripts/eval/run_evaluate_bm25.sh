#!/bin/bash
# Run BM25 baseline evaluation
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_DIR="$PROJECT_DIR/data/eval"
OUTPUT_DIR="$PROJECT_DIR/results/bm25_baseline"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Running BM25 baseline evaluation"
echo "=============================================="

# Dataset names (matching run_evaluate_distilled.sh)
DATASETS=(
    HC3Finance
    CUREv1_en
    AILA_casedocs
    legal_summarization
    LegalQuAD
    finqa
    financebench
    multi-cpr-video
)

python "$SCRIPT_DIR/evaluate_bm25.py" \
    --data-dir "$DATA_DIR" \
    --datasets "${DATASETS[@]}" \
    --output "$OUTPUT_DIR/bm25_results.json" \
    --batch-size 1000

echo ""
echo "=============================================="
echo "BM25 baseline evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
