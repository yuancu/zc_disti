#!/bin/bash
set -e

SCRIPT_DIR="scripts"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR = "${PROJECT_DIR}/data/finetune"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# space separated dataset names [data-dir/dataset-name]
DATASETS="sample"

for dataset in $DATASETS; do
    echo "===== Training: $dataset ====="
    accelerate launch --multi_gpu --num_processes=4 ${SCRIPT_DIR}/finetune.py \
        --data-dir DATA_DIR \
        --dataset $dataset \
        --output-dir ${PROJECT_DIR}/output \
        --max-seq-length 1024
done

echo "All done!"
