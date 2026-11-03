#!/bin/bash
#SBATCH --job-name=qd_cross_task
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm_cross_task_%j.log
#SBATCH --error=logs/slurm_cross_task_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
# Zero-shot: E1M1-trained policies on My Way Home and Deadly Corridor, no retraining.
python evaluate_cross_task.py pixel_e1m1_runs/cross_task_results.json \
  e1m1_classical=classical:pixel_e1m1_runs/imitation_classical_v2_seed0/best.pt \
  e1m1_linear=linear:pixel_e1m1_runs/imitation_linear_v2_seed0/best.pt \
  e1m1_circuit=quantum:pixel_e1m1_runs/imitation_quantum_v2_seed0/best.pt \
  e1m1_finetuned=classical:pixel_e1m1_runs/finetune_classical_richcritic_seed0/best.pt:rich
