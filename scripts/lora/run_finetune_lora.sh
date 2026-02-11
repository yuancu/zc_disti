#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# space separated dataset names [data-dir/dataset-name]
DATASETS="sample"

for dataset in $DATASETS; do
    echo "===== Training: $dataset ====="
    accelerate launch --multi_gpu --num_processes=4 "$SCRIPT_DIR/finetune_lora.py" \
        --data-dir "$PROJECT_DIR/synthetic_data/training-data-20k" \
        --dataset $dataset \
        --output-dir "$PROJECT_DIR/artifacts/models/lora" \
        --max-seq-length 1024
done

echo "All done!"
