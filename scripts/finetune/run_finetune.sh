#!/bin/bash
set -e

SCRIPT_DIR="scripts/finetune"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
DATA_DIR="${PROJECT_DIR}/data/finetune"

export NUM_DEVICES=1
export CUDA_VISIBLE_DEVICES=0

# space separated dataset names [data-dir/dataset-name]
DATASETS="financebench finqa"

for dataset in $DATASETS; do
    echo "===== Training: $dataset ====="
    accelerate launch --num_processes=$NUM_DEVICES ${SCRIPT_DIR}/finetune.py \
        --data-dir ${DATA_DIR} \
        --dataset $dataset \
        --output-dir ${PROJECT_DIR}/artifacts/models/finetuned \
        --log-dir ${PROJECT_DIR}/output \
        --max-seq-length 1024 \
        --batch-size 8
done

echo "All done!"
