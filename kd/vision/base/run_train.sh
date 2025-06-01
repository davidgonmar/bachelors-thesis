#!/usr/bin/env bash
PARALLELISM="${1:-8}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR"

SEEDS=(0 1 2 3 4)
ALPHAS=(0.8 0.9 0.95 0.98)

CMDS=()
for alpha in "${ALPHAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    # include both seed and alpha in filename
    out="${DIR}/train_s${seed}_a${alpha}.json"
    CMDS+=("python \"$CURDIR/train.py\" \
      --nbits 2 \
      --epochs 200 \
      --batch_size 128 \
      --val_batch_size 512 \
      --pretrained_path resnet20.pth \
      --student_batches 1 \
      --matchers_batches 1 \
      --alpha \"$alpha\" \
      --seed \"$seed\" \
      --output_path \"$out\"")
  done
done

# run them in parallel
printf '%s\n' "${CMDS[@]}" | xargs -P "$PARALLELISM" -I CMD bash -c CMD
