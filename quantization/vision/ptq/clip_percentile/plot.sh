#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m quantization.vision.ptq.clip_percentile.plot \
  --json_path "$DIR/results/results.json" \
  --output_path "$DIR/results/plot.pdf" \