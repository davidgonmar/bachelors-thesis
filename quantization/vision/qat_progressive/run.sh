#!/usr/bin/env bash
PARALLELISM="${1:-4}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR"

METHODS=(lsq qat)
BITS_LISTS=("8 4 2" "8 4")
MILESTONES_LISTS=("25 50" "25")
SEEDS=(0 1 2 3 4)

CMDS=()

for method in "${METHODS[@]}"; do
  for i in "${!BITS_LISTS[@]}"; do
    bits="${BITS_LISTS[$i]}"
    milestones="${MILESTONES_LISTS[$i]}"
    bits_tag=${bits// /-}
    for seed in "${SEEDS[@]}"; do
      tag="${method}_p${bits_tag}_s${seed}"
      out="${DIR}/${tag}.json"
      CMDS+=("python -m quantization.vision.qat_progressive.qat_progressive \
--method $method --bits_list $bits --epoch_milestones $milestones \
--seed $seed --output_path $out")
    done
  done
done

printf '%s
' "${CMDS[@]}" | xargs -P "$PARALLELISM" -I CMD bash -c CMD
