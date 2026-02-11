#!/bin/bash
# Run evaluations for all LoRA models and baseline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
DATA_DIR="$PROJECT_DIR/data/eval"
OUTPUT_DIR="$PROJECT_DIR/results/lora"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Running evaluations for all LoRA models"
echo "=============================================="

# Define model-to-dataset mappings
# Format: "model_path|dataset_name|display_name"
declare -a EVALUATIONS=(
    "artifacts/models/lora/aila-casedocs|AILA_casedocs|aila-casedocs"
    "artifacts/models/lora/financebench|financebench|financebench"
    "artifacts/models/lora/finqa|finqa|finqa"
    "artifacts/models/lora/hc3finance|HC3Finance|hc3finance"
    "artifacts/models/lora/legal-summ|legal_summarization|legal-summ"
    "artifacts/models/lora/legalquad|LegalQuAD|legalquad"
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

# Then evaluate each LoRA model
for eval_config in "${EVALUATIONS[@]}"; do
    IFS='|' read -r model_path dataset_name display_name <<< "$eval_config"

    echo ""
    echo "=============================================="
    echo "Evaluating LoRA model: $display_name"
    echo "=============================================="

    python scripts/eval/evaluate_model.py \
        --model-path "$model_path" \
        --data-dir "$DATA_DIR" \
        --datasets "$dataset_name" \
        --output "$OUTPUT_DIR/lora_${display_name}.json" \
        --batch-size 16
done

echo ""
echo "=============================================="
echo "All evaluations complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="

# Generate report
python "$SCRIPT_DIR/generate_report.py" --results-dir "$OUTPUT_DIR" --prefix lora
