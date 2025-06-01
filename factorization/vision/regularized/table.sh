#!/usr/bin/env bash

RESULTS_DIR="$(dirname "$0")/results"
OUTPUT_TEX="$RESULTS_DIR/accuracy_table.tex"

python "$(dirname "$0")/table.py" \
  --results_dir "$RESULTS_DIR" \
  --output "$OUTPUT_TEX"

echo "LaTeX table saved to $OUTPUT_TEX"