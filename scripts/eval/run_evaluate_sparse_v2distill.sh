#!/bin/bash
set -e

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate pytorch 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_DIR="$PROJECT_DIR/data/eval"
MODEL_DIR="$PROJECT_DIR/artifacts/models/distilled-sparse-v2distill"
OUTPUT_DIR="$PROJECT_DIR/results/distilled-sparse-v2distill"

BASELINE_MODEL="opensearch-project/opensearch-neural-sparse-encoding-v2-distill"

mkdir -p "$OUTPUT_DIR"

declare -a EVALUATIONS=(
    "aila-casedocs|AILA_casedocs|aila-casedocs"
    "cure_v1|CUREv1_en|cure_v1"
    "financebench|financebench|financebench"
    "finqa|finqa|finqa"
    "hc3finance|HC3Finance|hc3finance"
    "legal-summ|legal_summarization|legal-summ"
    "legalquad|LegalQuAD|legalquad"
)

# Baseline
echo "=============================================="
echo "Evaluating BASELINE sparse model ($BASELINE_MODEL)"
echo "=============================================="

DATASETS=""
for eval_config in "${EVALUATIONS[@]}"; do
    IFS='|' read -r model_subdir dataset_name display_name <<< "$eval_config"
    DATASETS="$DATASETS $dataset_name"
done

CUDA_VISIBLE_DEVICES=1 python "$SCRIPT_DIR/evaluate_model.py" --sparse \
    --model-path "$BASELINE_MODEL" \
    --data-dir "$DATA_DIR" \
    --datasets $DATASETS \
    --output "$OUTPUT_DIR/baseline_results.json" \
    --batch-size 16

# Distilled models
for eval_config in "${EVALUATIONS[@]}"; do
    IFS='|' read -r model_subdir dataset_name display_name <<< "$eval_config"

    echo ""
    echo "=============================================="
    echo "Evaluating distilled sparse model: $display_name"
    echo "=============================================="

    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT_DIR/evaluate_model.py" --sparse \
        --model-path "$MODEL_DIR/$model_subdir" \
        --data-dir "$DATA_DIR" \
        --datasets "$dataset_name" \
        --output "$OUTPUT_DIR/distilled_${display_name}.json" \
        --batch-size 16
done

echo ""
echo "=============================================="
echo "All sparse v2-distill evaluations complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
