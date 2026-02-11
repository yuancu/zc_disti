#!/bin/bash
# Run evaluations for all fine-tuned models and baseline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
DATA_DIR="$PROJECT_DIR/data/eval"
OUTPUT_DIR="$PROJECT_DIR/results/tuned"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Running evaluations for all models"
echo "=============================================="

# Define model-to-dataset mappings
# Format: "model_path|dataset_name|display_name"
declare -a EVALUATIONS=(
    "artifacts/models/tuned/aila-casedocs|AILA_casedocs|aila-casedocs"
    "artifacts/models/tuned/financebench|financebench|financebench"
    "artifacts/models/tuned/finqa|finqa|finqa"
    "artifacts/models/tuned/hc3finance|HC3Finance|hc3finance"
    "artifacts/models/tuned/legal-summ|legal_summarization|legal-summ"
    "artifacts/models/tuned/legalquad|LegalQuAD|legalquad"
)

cd "$PROJECT_DIR"

# First, evaluate baseline model on all datasets
echo ""
echo "=============================================="
echo "Evaluating BASELINE model (BAAI/bge-m3)"
echo "=============================================="

DATASETS=""
for eval_config in "${EVALUATIONS[@]}"; do
    IFS='|' read -r model_path dataset_name display_name <<< "$eval_config"
    DATASETS="$DATASETS $dataset_name"
done

python scripts/eval/evaluate_model.py \
    --model-path BAAI/bge-m3 \
    --data-dir "$DATA_DIR" \
    --datasets $DATASETS \
    --output "$OUTPUT_DIR/baseline_results.json" \
    --batch-size 16

# Then evaluate each fine-tuned model
for eval_config in "${EVALUATIONS[@]}"; do
    IFS='|' read -r model_path dataset_name display_name <<< "$eval_config"

    echo ""
    echo "=============================================="
    echo "Evaluating FINE-TUNED model: $display_name"
    echo "=============================================="

    python scripts/eval/evaluate_model.py \
        --model-path "$model_path" \
        --data-dir "$DATA_DIR" \
        --datasets "$dataset_name" \
        --output "$OUTPUT_DIR/finetuned_${display_name}.json" \
        --batch-size 16
done

echo ""
echo "=============================================="
echo "All evaluations complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="

# Generate report
python "$SCRIPT_DIR/generate_report.py" --results-dir "$OUTPUT_DIR"
