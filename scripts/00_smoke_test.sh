#!/bin/bash
# A few minutes on CPU: checks ViZDoom, the E1M1 map/route/distance field, the teacher,
# and runs a handful of PPO and DAgger updates for every head. Numbers here mean nothing.
source "$(dirname "$0")/common.sh"
S=$OUT/smoke
$PYTHON diagnose_pixel_e1m1.py
for task in navigation corridor e1m1; do
  for head in classical linear quantum; do
    run $S/ppo_${task}_$head train_pixel_nav.py --task $task --model $head --decisions 2048 \
      --num-envs 2 --steps 128 --eval-every 4 --eval-episodes 2 --final-episodes 2 --log-every 2 \
      --output $S/ppo_${task}_$head
  done
done
for head in classical quantum; do
  run $S/imitation_$head train_e1m1_imitation.py --model $head --num-envs 2 --iterations 1 \
    --initial-steps 2000 --iter-steps 1000 --initial-epochs 1 --epochs 1 --eval-episodes 2 \
    --final-episodes 2 --max-episode 200 --weight-power 0.75 --output $S/imitation_$head
done
echo "smoke test finished: outputs in $S"
