#!/usr/bin/env bash
set -euo pipefail

PARALLEL_N="${1:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
FINETUNED_DIR="$RESULTS_DIR/finetuned"
TRAIN_SCRIPT="$SCRIPT_DIR/finetune.py"

MODELS=(
  "resnet20_regularized_0.005/flops_0.20"
  "resnet20_regularized_0.005/flops_0.25"
  "resnet20_regularized_0.005/flops_0.30"
  "resnet20_regularized_0.003/flops_0.35"
  "resnet20_regularized_0.004/flops_0.40"
  "resnet20_regularized_0.003/flops_0.45"
  "resnet20_regularized_0.003/flops_0.50"

  "resnet20_regularized_0.005/params_0.20"
  "resnet20_regularized_0.005/params_0.25"
  "resnet20_regularized_0.005/params_0.30"
  "resnet20_regularized_0.003/params_0.35"
  "resnet20_regularized_0.004/params_0.40"
  "resnet20_regularized_0.003/params_0.45"
  "resnet20_regularized_0.003/params_0.50"
)

mkdir -p "$FINETUNED_DIR"

CMDS=()
for MODEL in "${MODELS[@]}"; do
  SAFE_NAME="${MODEL%.pth}"
  SAFE_NAME="${SAFE_NAME//\//_}"
  LOG_PATH="$FINETUNED_DIR/${SAFE_NAME}.json"
  PRETRAINED_PATH="$RESULTS_DIR/${MODEL}.pth"
  CMDS+=("python \"$TRAIN_SCRIPT\" --model_name resnet20 --pretrained_path \"$PRETRAINED_PATH\" --log_path \"$LOG_PATH\"")
done

printf '%s\n' "${CMDS[@]}" | xargs -P "$PARALLEL_N" -I CMD bash -c CMD
