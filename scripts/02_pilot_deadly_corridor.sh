#!/bin/bash
# Table 2, Deadly Corridor (skill 3). HEAD=classical|linear|quantum, SEEDS="0 1 2".
# The circuit run in the report is seed 0 only: 19,500 s toward 3M decisions, then
# --resume toward 6M decisions for another 19,500 s (about 11 h in total on a V100).
# NOTE: the E1M1 runs warm-start from the classical seed-0 encoder trained here.
source "$(dirname "$0")/common.sh"
HEAD=${HEAD:-classical}
if [ "$HEAD" = quantum ]; then
  for seed in ${SEEDS:-0}; do
    dir=$OUT/pixel_corridor_runs/quantum_seed$seed
    common=(--task corridor --model quantum --seed $seed --head-lr 0.003 --eval-every 250
            --eval-episodes 32 --final-episodes 100 --max-seconds ${MAX_SECONDS:-19500} --output $dir)
    run $dir train_pixel_nav.py "${common[@]}" --decisions 3000000
    cp $dir/train.log $dir/train_part1.log
    run $dir train_pixel_nav.py "${common[@]}" --decisions 6000000 --resume
    cat $dir/train_part1.log $dir/train.log > $dir/train_all.log && mv $dir/train_all.log $dir/train.log
  done
  exit 0
fi
for seed in ${SEEDS:-0 1 2}; do
  extra=()
  [ "$seed" = 0 ] && [ "$HEAD" = classical ] && extra=(--random-baseline)  # random + always-forward rows
  dir=$OUT/pixel_corridor_runs/${HEAD}_seed$seed
  run $dir train_pixel_nav.py --task corridor --model $HEAD --seed $seed --decisions 3000000 \
    --eval-every 250 --eval-episodes 32 --final-episodes 100 --max-seconds ${MAX_SECONDS:-6000} \
    "${extra[@]}" --output $dir
done
