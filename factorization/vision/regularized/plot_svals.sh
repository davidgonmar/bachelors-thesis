#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
PY_SCRIPT="${SCRIPT_DIR}/plot_svals.py"

shopt -s nullglob
for ckpt in "$RESULTS_DIR"/*.pth; do
    base="$(basename "$ckpt" .pth)"
    out_pdf="${RESULTS_DIR}/${base}.pdf"
    echo "Processing $ckpt to $out_pdf"
    python "$PY_SCRIPT" --pretrained_path "$ckpt" --output_plot "$out_pdf"
done

echo "Processing resnet20.pth to ${RESULTS_DIR}/resnet20.pdf"
python "$PY_SCRIPT" --pretrained_path "resnet20.pth" --output_plot "${RESULTS_DIR}/resnet20.pdf"
