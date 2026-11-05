#!/bin/bash
#SBATCH --job-name=qd_robustness
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_robustness_%j.log
#SBATCH --error=logs/slurm_robustness_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
# Memorisation tests on the E1M1 imitation students (no retraining), 100 episodes each.
python evaluate_e1m1_robustness.py pixel_e1m1_runs/robustness_results.json 100 \
  classical=classical:pixel_e1m1_runs/imitation_classical_v2_seed0/best.pt \
  linear=linear:pixel_e1m1_runs/imitation_linear_v2_seed0/best.pt \
  circuit=quantum:pixel_e1m1_runs/imitation_quantum_v2_seed0/best.pt
