#!/usr/bin/env bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

python3 "$SCRIPT_DIR/plot.py" \
    --input "$RESULTS_DIR/train_s0.json" \
    --output "$RESULTS_DIR/plot.pdf"
