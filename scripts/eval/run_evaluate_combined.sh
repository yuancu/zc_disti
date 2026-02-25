#!/bin/bash
# Run combined BM25 + semantic search evaluation
#
# Each dataset uses its own finetuned model from artifacts/models/finetuned/.
# Also evaluates baseline (BAAI/bge-m3) on all datasets for comparison.
#
# Environment variable overrides:
#   BASELINE_MODEL   - Baseline model path (default: BAAI/bge-m3)
#   NORMALIZATION    - min_max or l2 (default: min_max)
#   ALPHA            - BM25 weight, 0.0-1.0 (default: 0.5)
#   RETRIEVAL_DEPTH  - Top-K from each system (default: 1000)
#
# Usage:
#   bash run_evaluate_combined.sh
#   ALPHA=0.7 bash run_evaluate_combined.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_DIR="$PROJECT_DIR/data/eval"
OUTPUT_DIR="$PROJECT_DIR/results/combined-distilled"
MODEL_DIR="$PROJECT_DIR/artifacts/models/distilled"

# Configuration with environment variable overrides
BASELINE_MODEL="${BASELINE_MODEL:-BAAI/bge-m3}"
NORMALIZATION="${NORMALIZATION:-min_max}"
ALPHA="${ALPHA:-0.5}"
RETRIEVAL_DEPTH="${RETRIEVAL_DEPTH:-1000}"

COMBINATIONS=(arithmetic geometric harmonic)

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Running combined BM25 + semantic evaluation"
echo "  Model dir:     $MODEL_DIR"
echo "  Baseline:      $BASELINE_MODEL"
echo "  Normalization: $NORMALIZATION"
echo "  Combinations:  ${COMBINATIONS[*]}"
echo "  Alpha:         $ALPHA"
echo "  Retrieval K:   $RETRIEVAL_DEPTH"
echo "=============================================="

# Format: "dataset_name|model_subdir"
EVALUATIONS=(
    "HC3Finance|hc3finance"
    "CUREv1_en|cure_v1"
    "AILA_casedocs|aila-casedocs"
    "legal_summarization|legal-summ"
    "LegalQuAD|legalquad"
    "finqa|finqa"
    "financebench|financebench"
    "multi-cpr-video|multi-cpr-video"
)

# Collect all dataset names for baseline evaluation
DATASETS=""
for eval_config in "${EVALUATIONS[@]}"; do
    IFS='|' read -r dataset_name model_subdir <<< "$eval_config"
    DATASETS="$DATASETS $dataset_name"
done

for COMBINATION in "${COMBINATIONS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Combination method: $COMBINATION"
    echo "=============================================="

    # Evaluate baseline model on all datasets
    echo ""
    echo "  Evaluating BASELINE model ($BASELINE_MODEL)"

    OUTPUT_FILE="$OUTPUT_DIR/combined_baseline_${NORMALIZATION}_${COMBINATION}_alpha${ALPHA}.json"

    python "$SCRIPT_DIR/evaluate_combined.py" \
        --model-path "$BASELINE_MODEL" \
        --data-dir "$DATA_DIR" \
        --datasets $DATASETS \
        --normalization "$NORMALIZATION" \
        --combination "$COMBINATION" \
        --alpha "$ALPHA" \
        --retrieval-depth "$RETRIEVAL_DEPTH" \
        --output "$OUTPUT_FILE" \
        --tokenizer-batch-size 1000 \
        --batch-size 64

    echo "  Results saved to: $OUTPUT_FILE"

    # Evaluate each finetuned model on its dataset
    for eval_config in "${EVALUATIONS[@]}"; do
        IFS='|' read -r dataset_name model_subdir <<< "$eval_config"
        MODEL_PATH="$MODEL_DIR/$model_subdir"

        echo ""
        echo "  Evaluating FINETUNED model: $dataset_name  Model: $MODEL_PATH"

        OUTPUT_FILE="$OUTPUT_DIR/combined_${dataset_name}_${NORMALIZATION}_${COMBINATION}_alpha${ALPHA}.json"

        python "$SCRIPT_DIR/evaluate_combined.py" \
            --model-path "$MODEL_PATH" \
            --data-dir "$DATA_DIR" \
            --datasets "$dataset_name" \
            --normalization "$NORMALIZATION" \
            --combination "$COMBINATION" \
            --alpha "$ALPHA" \
            --retrieval-depth "$RETRIEVAL_DEPTH" \
            --output "$OUTPUT_FILE" \
            --tokenizer-batch-size 1000 \
            --batch-size 64

        echo "  Results saved to: $OUTPUT_FILE"
    done
done

echo ""
echo "=============================================="
echo "All combined evaluations complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
