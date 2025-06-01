#!/usr/bin/env bash
PARALLELISM="${1:-5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

TRAIN_SCRIPT="$SCRIPT_DIR/train.py"
PRETRAINED_PATH="resnet20.pth"

REG_WEIGHTS=(0.001 0.002 0.003 0.004 0.005)

CMDS=()

for weight in "${REG_WEIGHTS[@]}"; do
  LOG_PATH="$RESULTS_DIR/training_log_${weight}.json"
  SAVE_PATH="$RESULTS_DIR/resnet20_regularized_${weight}.pth"

  CMDS+=("python \"$TRAIN_SCRIPT\" \
--log_path \"$LOG_PATH\" \
--save_path \"$SAVE_PATH\" \
--pretrained_path \"$PRETRAINED_PATH\" \
--epochs 200 \
--batch_size 128 \
--lr 0.01 \
--step_size 80 \
--gamma 0.1 \
--reg_weight $weight")
done

# Run in parallel
printf '%s\n' "${CMDS[@]}" | xargs -P "$PARALLELISM" -I CMD bash -c CMD
