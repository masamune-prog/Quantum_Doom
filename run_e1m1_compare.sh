#!/bin/bash
#SBATCH --job-name=qd_e1m1_compare
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_e1m1_compare_%j.log
#SBATCH --error=logs/slurm_e1m1_compare_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
# 300 fresh held-out seeds (30000-30299), disjoint from selection (10000+) and the earlier 100.
python evaluate_e1m1_checkpoints.py pixel_e1m1_runs/compare_imitation_vs_finetune.json 30000 300 \
  imitation=pixel_e1m1_runs/imitation_classical_v2_seed0/best.pt \
  finetuned=pixel_e1m1_runs/finetune_classical_richcritic_seed0/best.pt:rich
