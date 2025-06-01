#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINEAR_RESULTS="$DIR/results/linear"
EXP_RESULTS="$DIR/results/exp"
TABLE_FILE="$DIR/results/last_accuracy.tex"

python "$DIR/table.py" --results-dirs "$LINEAR_RESULTS" "$EXP_RESULTS" --output "$TABLE_FILE"
echo "[make_table.sh] LaTeX table written to $TABLE_FILE"