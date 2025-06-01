#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FLOPS_JSONS=(
  "$DIR/results/global_factorization_resnet20_regularized_0.001_flops.json"
  "$DIR/results/global_factorization_resnet20_regularized_0.002_flops.json"
  "$DIR/results/global_factorization_resnet20_regularized_0.003_flops.json"
  "$DIR/results/global_factorization_resnet20_regularized_0.004_flops.json"
  "$DIR/results/global_factorization_resnet20_regularized_0.005_flops.json"
)

PARAMS_JSONS=(
  "$DIR/results/global_factorization_resnet20_regularized_0.001_params.json"
  "$DIR/results/global_factorization_resnet20_regularized_0.002_params.json"
  "$DIR/results/global_factorization_resnet20_regularized_0.003_params.json"
  "$DIR/results/global_factorization_resnet20_regularized_0.004_params.json"
  "$DIR/results/global_factorization_resnet20_regularized_0.005_params.json"
)

python "$DIR/graph.py" \
  --flops-jsons "${FLOPS_JSONS[@]}" \
  --params-jsons "${PARAMS_JSONS[@]}" \
  --output-flops "$DIR/results/flops_vs_acc.pdf" \
  --output-params "$DIR/results/params_vs_acc.pdf"