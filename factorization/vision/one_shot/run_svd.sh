#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

PRETRAINED_RESNET20="resnet20.pth"

# global activation-aware
python "${SCRIPT_DIR}/svd_global_activation_aware.py" \
  --model_name resnet20 \
  --pretrained_path "${PRETRAINED_RESNET20}" \
  --metric flops \
  --output_file "${RESULTS_DIR}/global_activation_aware_flops_results.json"

python "${SCRIPT_DIR}/svd_global_activation_aware.py" \
  --model_name resnet20 \
  --pretrained_path "${PRETRAINED_RESNET20}" \
  --metric params \
  --output_file "${RESULTS_DIR}/global_activation_aware_params_results.json"

# global regular
python "${SCRIPT_DIR}/svd_global.py" \
  --model_name resnet20 \
  --pretrained_path "${PRETRAINED_RESNET20}" \
  --metric flops \
  --output_file "${RESULTS_DIR}/global_flops_results.json"

python "${SCRIPT_DIR}/svd_global.py" \
  --model_name resnet20 \
  --pretrained_path "${PRETRAINED_RESNET20}" \
  --metric params \
  --output_file "${RESULTS_DIR}/global_params_results.json"

# manual activation-aware
python "${SCRIPT_DIR}/svd_manual_activation_aware.py" \
  --model_name resnet20 \
  --pretrained_path "${PRETRAINED_RESNET20}" \
  --metric energy \
  --output_file "${RESULTS_DIR}/manual_activation_aware_energy_results.json"

python "${SCRIPT_DIR}/svd_manual_activation_aware.py" \
  --model_name resnet20 \
  --pretrained_path "${PRETRAINED_RESNET20}" \
  --metric rank \
  --output_file "${RESULTS_DIR}/manual_activation_aware_rank_results.json"

# manual regular
python "${SCRIPT_DIR}/svd_manual.py" \
  --model_name resnet20 \
  --pretrained_path "${PRETRAINED_RESNET20}" \
  --metric energy \
  --output_file "${RESULTS_DIR}/manual_energy_results.json"

python "${SCRIPT_DIR}/svd_manual.py" \
  --model_name resnet20 \
  --pretrained_path "${PRETRAINED_RESNET20}" \
  --metric rank \
  --output_file "${RESULTS_DIR}/manual_rank_results.json"
