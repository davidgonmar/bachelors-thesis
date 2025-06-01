#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROG_DIR="$SCRIPT_DIR/qat_progressive/results"
QAT_DIR="$SCRIPT_DIR/qat/results"
OUT_PDF="$SCRIPT_DIR/results/compare.pdf"
mkdir -p "$SCRIPT_DIR/results"

python3 "$SCRIPT_DIR/plot_qat_vs_progressive.py" \
  --prog_dir "$PROG_DIR" \
  --static_dir  "$QAT_DIR" \
  --out_pdf  "$OUT_PDF"