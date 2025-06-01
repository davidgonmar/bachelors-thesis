#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR/results"
python -m quantization.vision.ptq.non_symmetric.quant \
  --model_name resnet20 \
  --pretrained_path resnet20.pth \
  --seed 0 \
  --runs 5 \
  --output_path "$DIR/results/results.json"