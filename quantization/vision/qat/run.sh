#!/usr/bin/env bash
PARALLELISM="${1:-6}"

# directory that will hold one JSON file per run
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$DIR"

METHODS=(lsq qat)
BITS=(2 4)
SEEDS=(0 1 2 3 4)

CMDS=()

for method in "${METHODS[@]}"; do
  for w in "${BITS[@]}"; do
    for a in "${BITS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        tag="${method}_w${w}a${a}_s${seed}"
        out="${DIR}/${tag}.json"
        CMDS+=("python \"$CURDIR/qat.py\" \
--method \"$method\" --nbits_activations \"$a\" --nbits_weights \"$w\" \
--seed \"$seed\" --output_path \"$out\"")
      done
    done
  done
done

printf '%s\n' "${CMDS[@]}" | xargs -P "$PARALLELISM" -I CMD bash -c CMD