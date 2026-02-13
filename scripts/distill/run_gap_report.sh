#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

TEACHER_DIR="$PROJECT_DIR/artifacts/teacher-score-gemma2"
STUDENT_DIR="$PROJECT_DIR/artifacts/student-score-bge-m3"
OUTPUT="$PROJECT_DIR/docs/teacher-student-gap.md"

python "$SCRIPT_DIR/generate_gap_report.py" \
  --teacher_dir "$TEACHER_DIR" \
  --student_dir "$STUDENT_DIR" \
  --output "$OUTPUT"
