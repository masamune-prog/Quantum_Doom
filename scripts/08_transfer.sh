#!/bin/bash
# Section 4.4, "Do the students memorise the level?":
#  (a) E1M1 students run zero-shot on My Way Home and Deadly Corridor (6-10 and 0-1 exits)
#  (b) the pilot-trained models run zero-shot on E1M1 (no progress)
source "$(dirname "$0")/common.sh"
mkdir -p $OUT/pixel_e1m1_runs
$PYTHON evaluate_cross_task.py $OUT/pixel_e1m1_runs/cross_task_results.json \
  e1m1_classical=classical:$(ckpt pixel_e1m1_runs/imitation_classical_v2_seed0) \
  e1m1_linear=linear:$(ckpt pixel_e1m1_runs/imitation_linear_v2_seed0) \
  e1m1_circuit=quantum:$(ckpt pixel_e1m1_runs/imitation_quantum_v2_seed0) \
  e1m1_finetuned=classical:$(ckpt pixel_e1m1_runs/finetune_classical_richcritic_seed0):rich \
  | tee $OUT/pixel_e1m1_runs/cross_task.log
$PYTHON evaluate_specialists_on_e1m1.py $OUT/pixel_e1m1_runs/specialists_on_e1m1.json \
  classical:$(ckpt pixel_nav_runs/classical_seed0):$(ckpt pixel_corridor_runs/classical_seed0) \
  quantum:$(ckpt pixel_nav_runs/quantum_seed0):$(ckpt pixel_corridor_runs/quantum_seed0) \
  | tee $OUT/pixel_e1m1_runs/specialists_on_e1m1.log
