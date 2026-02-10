#!/bin/bash
# Run evaluations for distilled models
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="/home/ubuntu/src/ir-fine-tune-evaluation/data"
MODEL_DIR="$PROJECT_DIR/artifacts/distilled"
OUTPUT_DIR="$PROJECT_DIR/results"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Running evaluations for distilled models"
echo "=============================================="

# Format: "model_subdir|dataset_name|display_name"
declare -a EVALUATIONS=(
    "bge-m3_hc3finance|HC3Finance|hc3finance"
    # "bge-m3_cure_v1|CUREv1_en|cure_v1"
)

for eval_config in "${EVALUATIONS[@]}"; do
    IFS='|' read -r model_subdir dataset_name display_name <<< "$eval_config"

    echo ""
    echo "=============================================="
    echo "Evaluating distilled model: $display_name"
    echo "=============================================="

    python "$SCRIPT_DIR/evaluate_model.py" \
        --model-path "$MODEL_DIR/$model_subdir" \
        --data-dir "$DATA_DIR" \
        --datasets "$dataset_name" \
        --output "$OUTPUT_DIR/distilled_${display_name}.json" \
        --batch-size 64
done

echo ""
echo "=============================================="
echo "All distilled evaluations complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
