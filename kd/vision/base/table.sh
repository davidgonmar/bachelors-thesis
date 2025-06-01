#!/usr/bin/env bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

python3 "$SCRIPT_DIR/table.py" \
    --dir "$RESULTS_DIR" \
    --output "$RESULTS_DIR/table.tex"
