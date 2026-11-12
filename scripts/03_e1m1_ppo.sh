#!/bin/bash
# Table 3, PPO on E1M1 (classical head). ATTEMPT=1|2|3|4naive|4kl1|4kl01|4kill8|4rich
#   1  reverse-start curriculum, corridor-encoder warm start, 10M decisions      (~2.2 h, V100)
#   2  uniform route starts, no stages (cancelled at ~3M decisions in the report)
#   3  uniform starts + route-direction/progress auxiliary targets, 10M decisions
#   4* PPO fine-tuning of the DAgger classical student (needs 05_e1m1_imitation.sh first,
#      or CKPT=pretrained); 4rich is the run evaluated on 300 seeds by 06_*.sh
# Attempts 1-2 ran before the route auxiliary targets existed, hence --no-route-aux.
source "$(dirname "$0")/common.sh"
ATTEMPT=${ATTEMPT:-1}
ENC=${INIT_ENCODER:-$(ckpt pixel_corridor_runs/classical_seed0)}
ppo=(--task e1m1 --model classical --seed 0 --eval-every 250 --eval-episodes 32
     --final-episodes 100 --max-seconds ${MAX_SECONDS:-19500})
case $ATTEMPT in
  1) dir=$OUT/pixel_e1m1_runs/classical_warm_reverse_seed0
     run $dir train_pixel_nav.py "${ppo[@]}" --decisions 10000000 --curriculum reverse \
       --no-route-aux --random-baseline --init-encoder "$ENC" --output $dir ;;
  2) dir=$OUT/pixel_e1m1_runs/classical_warm_uniform_noroute_seed0
     run $dir train_pixel_nav.py "${ppo[@]}" --decisions ${DECISIONS:-3000000} --curriculum uniform \
       --no-route-aux --init-encoder "$ENC" --output $dir ;;
  3) dir=$OUT/pixel_e1m1_runs/classical_warm_uniform_route_seed0
     run $dir train_pixel_nav.py "${ppo[@]}" --decisions 10000000 --curriculum uniform \
       --random-baseline --init-encoder "$ENC" --output $dir ;;
  4*) INIT=${INIT:-$(ckpt pixel_e1m1_runs/imitation_classical_v2_seed0)}
     ft=(--task e1m1 --model classical --seed 0 --decisions 3000000 --init-checkpoint "$INIT"
         --value-warmup 100 --lr 5e-5 --head-lr 1e-4 --entropy-coef 0.005 --curriculum uniform
         --real-fraction 0.6 --eval-every 50 --eval-episodes 64 --final-episodes 100
         --max-seconds ${MAX_SECONDS:-19500})
     case $ATTEMPT in
       4naive) name=collapsed;     extra=() ;;
       4kl1)   name=anchored_kl1;  extra=(--freeze-encoder --anchor-kl-coef 1.0 --head-lr 5e-5) ;;
       4kl01)  name=loose_kl01;    extra=(--freeze-encoder --anchor-kl-coef 0.1 --head-lr 2e-4 --entropy-coef 0.01) ;;
       4kill8) name=kill8;         extra=(--freeze-encoder --anchor-kl-coef 0.1 --head-lr 2e-4 --entropy-coef 0.01 --kill-weight 8) ;;
       4rich)  name=richcritic;    extra=(--freeze-encoder --anchor-kl-coef 0.1 --head-lr 2e-4 --entropy-coef 0.01 --rich-critic) ;;
       *) echo "unknown ATTEMPT=$ATTEMPT" >&2; exit 1 ;;
     esac
     dir=$OUT/pixel_e1m1_runs/finetune_classical_${name}_seed0
     run $dir train_pixel_nav.py "${ft[@]}" "${extra[@]}" --output $dir ;;
  *) echo "unknown ATTEMPT=$ATTEMPT" >&2; exit 1 ;;
esac
