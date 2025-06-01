#!/usr/bin/env bash
set -euo pipefail

# Directory this script lives in
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Plot linear-scheduler results into its own folder
python -m sparsity.vision.structured_iterative.plot \
  "$DIR/results/linear" \
  "$DIR/results/linear/pruning_accuracy.pdf"
echo "[plot_results.sh] Linear plot written to results/linear/pruning_accuracy.pdf"

# Plot exp-scheduler results into its own folder
python -m sparsity.vision.structured_iterative.plot \
  "$DIR/results/exp" \
  "$DIR/results/exp/pruning_accuracy.pdf"
echo "[plot_results.sh] Exp plot written to results/exp/pruning_accuracy.pdf"