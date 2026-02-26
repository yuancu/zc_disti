#!/bin/bash
# Run evaluations for distilled sparse models
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_DIR="$PROJECT_DIR/data/eval"
MODEL_DIR="$PROJECT_DIR/artifacts/models/distilled-sparse-gte-500steps-seeded"
OUTPUT_DIR="$PROJECT_DIR/results/distilled-sparse-gte-500steps-seeded"

BASELINE_MODEL="opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"
# BASELINE_MODEL="opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Running evaluations for distilled sparse models"
echo "=============================================="

# Format: "model_subdir|dataset_name|display_name"
declare -a EVALUATIONS=(
    "aila-casedocs|AILA_casedocs|aila-casedocs"
    "cure_v1|CUREv1_en|cure_v1"
    "financebench|financebench|financebench"
    "finqa|finqa|finqa"
    "hc3finance|HC3Finance|hc3finance"
    "legal-summ|legal_summarization|legal-summ"
    "legalquad|LegalQuAD|legalquad"
    "multi-cpr-ecom|multi-cpr-ecom|multi_cpr_ecom"
    "multi-cpr-medical|multi-cpr-medical|multi_cpr_medical"
)

# First, evaluate baseline model on all datasets
echo ""
echo "=============================================="
echo "Evaluating BASELINE sparse model ($BASELINE_MODEL)"
echo "=============================================="

DATASETS=""
for eval_config in "${EVALUATIONS[@]}"; do
    IFS='|' read -r model_subdir dataset_name display_name <<< "$eval_config"
    DATASETS="$DATASETS $dataset_name"
done

python "$SCRIPT_DIR/evaluate_model.py" --sparse \
    --model-path "$BASELINE_MODEL" \
    --data-dir "$DATA_DIR" \
    --datasets $DATASETS \
    --output "$OUTPUT_DIR/baseline_results.json" \
    --batch-size 16

# Then evaluate each distilled sparse model
for eval_config in "${EVALUATIONS[@]}"; do
    IFS='|' read -r model_subdir dataset_name display_name <<< "$eval_config"

    echo ""
    echo "=============================================="
    echo "Evaluating distilled sparse model: $display_name"
    echo "=============================================="

    python "$SCRIPT_DIR/evaluate_model.py" --sparse \
        --model-path "$MODEL_DIR/$model_subdir" \
        --data-dir "$DATA_DIR" \
        --datasets "$dataset_name" \
        --output "$OUTPUT_DIR/distilled_${display_name}_gemma2_500steps.json" \
        --batch-size 16
done

echo ""
echo "=============================================="
echo "All sparse evaluations complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="

# Generate report (reuses the same report generator -- output JSON format is identical)
python "$SCRIPT_DIR/generate_report.py" --results-dir "$OUTPUT_DIR" --prefix distilled
