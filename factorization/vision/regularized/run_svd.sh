#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
FACTORIZE_SCRIPT="$SCRIPT_DIR/svd_global.py"

# loop over every model checkpoint
for MODEL_PATH in "$RESULTS_DIR"/resnet20_regularized_*.pth; do
  MODEL_FILE="$(basename "$MODEL_PATH")"
  BASE_NAME="${MODEL_FILE%.pth}"

  # for each metric, run factorization and tag output
  for METRIC in params flops; do
    OUTPUT_FILE="$RESULTS_DIR/global_factorization_${BASE_NAME}_${METRIC}.json"
    echo "Running factorization on $MODEL_FILE with metric=$METRIC"
    python "$FACTORIZE_SCRIPT" \
      --pretrained_path "$MODEL_PATH" \
      --metric "$METRIC" \
      --output_file "$OUTPUT_FILE"
  done
done
