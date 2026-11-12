#!/bin/bash
# Table 4, E1M1 students that imitate the scripted teacher. VARIANT=
#   classical | linear | quantum   DAgger: 200K teacher states + 8 rounds x 60K student states
#   bc200k | bc680k                behaviour cloning only (classical head, no student rollouts)
# All warm-start the CNN encoder from the Deadly Corridor classical seed-0 run.
# Wall time on a V100: ~25 min (classical/linear), ~1 h 53 min (circuit).
source "$(dirname "$0")/common.sh"
VARIANT=${VARIANT:-classical}
ENC=${INIT_ENCODER:-$(ckpt pixel_corridor_runs/classical_seed0)}
base=(--seed 0 --init-encoder "$ENC" --weight-power 0.75)
case $VARIANT in
  classical|linear) name=imitation_${VARIANT}_v2_seed0; args=(--model $VARIANT) ;;
  # The circuit used 3 epochs in round 0 and 2 per round after (classical: 6 and 3) to fit its budget.
  quantum) name=imitation_quantum_v2_seed0
           args=(--model quantum --head-lr 0.003 --initial-epochs 3 --epochs 2) ;;
  bc200k)  name=imitation_classical_bc200k_seed0; args=(--model classical --iterations 0) ;;
  bc680k)  name=imitation_classical_bc680k_seed0
           args=(--model classical --iterations 0 --initial-steps 680000) ;;
  *) echo "unknown VARIANT=$VARIANT" >&2; exit 1 ;;
esac
dir=$OUT/pixel_e1m1_runs/$name
run $dir train_e1m1_imitation.py "${base[@]}" "${args[@]}" --output $dir
