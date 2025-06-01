#!/usr/bin/env bash
PARALLELISM="${1:-1}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR"

SEED=0

CMD="python \"$CURDIR/train.py\" \
  --nbits 2 \
  --epochs 60 \
  --batch_size 128 \
  --val_batch_size 512 \
  --pretrained_path resnet20.pth \
  --student_batches 1 \
  --matchers_batches 1 \
  --seed \"$SEED\" \
  --output_path \"$DIR/train_s${SEED}.json\""

printf '%s\n' "$CMD" | xargs -P "$PARALLELISM" -I CMD bash -c CMD