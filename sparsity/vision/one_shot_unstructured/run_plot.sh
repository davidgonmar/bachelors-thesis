#!/usr/bin/env bash
set -euo pipefail

# Directory of this script
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESULTS_DIR="$DIR/results"
OUT="$RESULTS_DIR/one_shot_unstructured_pruning_cifar10.pdf"

python -m sparsity.vision.one_shot_unstructured.plot "$RESULTS_DIR" --outfile "$OUT"

echo "Plot saved to $OUT"
