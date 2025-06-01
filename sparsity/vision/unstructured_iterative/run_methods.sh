#!/usr/bin/env bash
set -euo pipefail

PARALLELISM="${1:-6}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$DIR/results"
mkdir -p "$RESULTS_DIR"


SCHEDULERS=(linear exp)
METHODS=(taylor magnitude_activations magnitude_weights)


MODEL="resnet20"
PRETRAINED="resnet20.pth"
BATCH_SIZE=128
TEST_BATCH=512
WORKERS=4
LR=0.001
MOMENTUM=0.9
WD=5e-4
TARGET_SPARSITY=0.80
N_ITERS=20
EPOCHS_PER_ITER=10
CALIBRATION_SAMPLES=512
CALIBRATION_BS=4
SEED=0


CMDS=()

for sched in "${SCHEDULERS[@]}"; do
  mkdir -p "$RESULTS_DIR/$sched"
  for m in "${METHODS[@]}"; do
    out="$RESULTS_DIR/${sched}/${m}_stats.json"
    CMDS+=("python -m sparsity.vision.unstructured_iterative.unstructured_iterative_pruning_cifar10 \
      --model \"$MODEL\" \
      --pretrained_path \"$PRETRAINED\" \
      --batch_size \"$BATCH_SIZE\" \
      --test_batch_size \"$TEST_BATCH\" \
      --train_workers \"$WORKERS\" \
      --test_workers \"$WORKERS\" \
      --lr \"$LR\" \
      --momentum \"$MOMENTUM\" \
      --weight_decay \"$WD\" \
      --target_sparsity \"$TARGET_SPARSITY\" \
      --n_iters \"$N_ITERS\" \
      --epochs_per_iter \"$EPOCHS_PER_ITER\" \
      --scheduler \"$sched\" \
      --method \"$m\" \
      --calibration_samples \"$CALIBRATION_SAMPLES\" \
      --calibration_bs \"$CALIBRATION_BS\" \
      --seed \"$SEED\" \
      --stats_file \"$out\"")
  done
done


printf '%s\n' "${CMDS[@]}" | xargs -P "$PARALLELISM" -I CMD bash -c CMD
