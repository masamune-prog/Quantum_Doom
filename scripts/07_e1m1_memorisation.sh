#!/bin/bash
# Table 5: perturbation tests of the three DAgger students (no retraining), 100 episodes per
# cell on seeds 40000-40099, episodes cut at 800 decisions. Also runs the teacher rows.
source "$(dirname "$0")/common.sh"
mkdir -p $OUT/pixel_e1m1_runs
$PYTHON evaluate_e1m1_robustness.py $OUT/pixel_e1m1_runs/robustness_results.json 100 \
  classical=classical:$(ckpt pixel_e1m1_runs/imitation_classical_v2_seed0) \
  linear=linear:$(ckpt pixel_e1m1_runs/imitation_linear_v2_seed0) \
  circuit=quantum:$(ckpt pixel_e1m1_runs/imitation_quantum_v2_seed0) \
  | tee $OUT/pixel_e1m1_runs/robustness.log
