#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR/results"

MODEL="resnet20"
PRETRAINED="resnet20.pth"
BATCH_SIZE=128
WORKERS=4
CALIBRATION_SAMPLES=512
CALIBRATION_BS=4
SEED=0

METHODS=(taylor_no_bias norm_activations taylor_bias norm_weights)
TARGETS=(0.00 0.025 0.05 0.075 0.1 0.125 0.15)

for m in "${METHODS[@]}"; do
  for t in "${TARGETS[@]}"; do
    tag="${m}_$(printf '%.2f' "$t")"
    echo "=== $m | target $t ==="
    python -m sparsity.vision.one_shot_structured.one_shot_structured_pruning_cifar10 \
      --model "$MODEL" \
      --pretrained_path "$PRETRAINED" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$WORKERS" \
      --target_sparsity "$t" \
      --method "$m" \
      --calibration_samples "$CALIBRATION_SAMPLES" \
      --calibration_bs "$CALIBRATION_BS" \
      --seed "$SEED" \
      --stats_file "$DIR/results/${tag}_stats.json"
  done
done