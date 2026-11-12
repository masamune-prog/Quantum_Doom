#!/bin/bash
# Figure 1 (RL training process) and the circuit drawing behind Figure 2, plus one gameplay
# video of a successful held-out episode for each E1M1 student.
source "$(dirname "$0")/common.sh"
$PYTHON draw_rl_diagram.py $OUT/figures
$PYTHON draw_circuit.py $OUT/figures
$PYTHON record_gameplay.py --out $OUT/videos \
  $(ckpt pixel_e1m1_runs/imitation_classical_v2_seed0) \
  $(ckpt pixel_e1m1_runs/imitation_linear_v2_seed0) \
  $(ckpt pixel_e1m1_runs/imitation_quantum_v2_seed0)
