#!/bin/bash
# Table 3, attempt 4: imitation student vs best PPO fine-tuned checkpoint on 300 fresh seeds
# (30000-30299): 238 vs 255 greedy exits in the report.
source "$(dirname "$0")/common.sh"
mkdir -p $OUT/pixel_e1m1_runs
$PYTHON evaluate_e1m1_checkpoints.py $OUT/pixel_e1m1_runs/compare_imitation_vs_finetune.json 30000 300 \
  imitation=$(ckpt pixel_e1m1_runs/imitation_classical_v2_seed0) \
  finetuned=$(ckpt pixel_e1m1_runs/finetune_classical_richcritic_seed0):rich \
  | tee $OUT/pixel_e1m1_runs/compare_imitation_vs_finetune.log
