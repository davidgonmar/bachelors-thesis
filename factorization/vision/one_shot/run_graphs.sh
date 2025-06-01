#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"



python "${SCRIPT_DIR}/graph.py" \
  --global_flops_json "${RESULTS_DIR}/global_flops_results.json" \
  --global_params_json "${RESULTS_DIR}/global_params_results.json" \
  --global_activation_aware_flops_json "${RESULTS_DIR}/global_activation_aware_flops_results.json" \
  --global_activation_aware_params_json "${RESULTS_DIR}/global_activation_aware_params_results.json" \
  --manual_rank_json "${RESULTS_DIR}/manual_rank_results.json" \
  --manual_energy_json "${RESULTS_DIR}/manual_energy_results.json" \
  --manual_activation_aware_rank_json "${RESULTS_DIR}/manual_activation_aware_rank_results.json" \
  --manual_activation_aware_energy_json "${RESULTS_DIR}/manual_activation_aware_energy_results.json" \
  --output_flops "${RESULTS_DIR}/flops_vs_acc.pdf" \
  --output_params "${RESULTS_DIR}/params_vs_acc.pdf"
