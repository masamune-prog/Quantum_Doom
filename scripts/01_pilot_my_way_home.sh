#!/bin/bash
# Table 2, My Way Home. HEAD=classical|linear|quantum, SEEDS="0 1 2".
# Classical/linear: ~31 min per seed on a V100. Circuit: stopped by a 19,500 s budget
# (~2.8-3.0M decisions on a V100); a faster GPU gets further in the same budget.
source "$(dirname "$0")/common.sh"
HEAD=${HEAD:-classical}
for seed in ${SEEDS:-0 1 2}; do
  extra=()
  [ "$seed" = 0 ] && [ "$HEAD" = classical ] && extra=(--random-baseline)  # random-action row
  if [ "$HEAD" = quantum ]; then
    extra+=(--head-lr 0.003 --max-seconds ${MAX_SECONDS:-19500})
  else
    extra+=(--max-seconds ${MAX_SECONDS:-6000})
  fi
  dir=$OUT/pixel_nav_runs/${HEAD}_seed$seed
  run $dir train_pixel_nav.py --task navigation --model $HEAD --seed $seed --decisions 3000000 \
    --eval-every 250 --eval-episodes 32 --final-episodes 100 "${extra[@]}" --output $dir
done
