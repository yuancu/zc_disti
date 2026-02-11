#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_DIR="$PROJECT_DIR/artifacts/hard-negative"
OUTPUT_DIR="$PROJECT_DIR/artifacts/teacher-score-gemma2"
NUM_GPUS=4

# DATASETS=("hc3finance" "cure_v1")
# DATASETS=("aila-casedocs" "legal-summ" "legalquad")
DATASETS=("finqa" "financebench")

mkdir -p "$OUTPUT_DIR"

for dataset in "${DATASETS[@]}"; do
  echo "========================================"
  echo "Processing: $dataset on $NUM_GPUS GPUs"
  echo "========================================"

  INPUT_FILE="$INPUT_DIR/${dataset}.jsonl"
  TOTAL_LINES=$(wc -l < "$INPUT_FILE")
  LINES_PER_CHUNK=$(( (TOTAL_LINES + NUM_GPUS - 1) / NUM_GPUS ))

  echo "Total lines: $TOTAL_LINES, ~$LINES_PER_CHUNK per GPU"

  WORK_DIR=$(mktemp -d)

  # Split input into chunks
  split -l "$LINES_PER_CHUNK" -d -a 1 "$INPUT_FILE" "$WORK_DIR/chunk_"

  # Launch one process per GPU
  PIDS=()
  for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    CHUNK="$WORK_DIR/chunk_${gpu_id}"
    if [ ! -f "$CHUNK" ]; then
      # Fewer chunks than GPUs (small dataset)
      continue
    fi
    echo "  GPU $gpu_id: $(wc -l < "$CHUNK") lines"
    CUDA_VISIBLE_DEVICES="$gpu_id" python "$SCRIPT_DIR/teacher_score.py" \
      --model gemma2 \
      --max_length 512 \
      --batch_size 32 \
      --input "$CHUNK" \
      --output "$WORK_DIR/out_${gpu_id}.json" \
      > "$WORK_DIR/log_${gpu_id}.txt" 2>&1 &
    PIDS+=("$gpu_id:$!")
  done

  # Wait and check for failures
  FAILED=0
  for entry in "${PIDS[@]}"; do
    gpu_id="${entry%%:*}"
    pid="${entry##*:}"
    if ! wait "$pid"; then
      echo "  GPU $gpu_id FAILED. Log:"
      cat "$WORK_DIR/log_${gpu_id}.txt"
      FAILED=1
    fi
  done

  if [ "$FAILED" -eq 1 ]; then
    echo "Some shards failed for $dataset. Aborting."
    rm -rf "$WORK_DIR"
    exit 1
  fi

  # Merge shard outputs into a single JSON array
  python -c "
import json, glob, sys
merged = []
for f in sorted(glob.glob('$WORK_DIR/out_[0-9].json')):
    with open(f) as fh:
        merged.extend(json.load(fh))
with open('$OUTPUT_DIR/${dataset}.json', 'w') as fh:
    json.dump(merged, fh)
print(f'Merged {len(merged)} samples into $OUTPUT_DIR/${dataset}.json')
"

  rm -rf "$WORK_DIR"
  echo ""
done

echo "Done. Results saved to: $OUTPUT_DIR"
