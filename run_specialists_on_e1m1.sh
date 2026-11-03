#!/bin/bash
#SBATCH --job-name=qd_specialists_e1m1
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_specialists_e1m1_%j.log
#SBATCH --error=logs/slurm_specialists_e1m1_%j.err
set -euo pipefail
cd /tc1home/UG/zgoh033/quantum_doom
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
# Zero-shot: the My Way Home and Deadly Corridor specialists (classical, then circuit) on E1M1.
python evaluate_specialists_on_e1m1.py pixel_e1m1_runs/specialists_on_e1m1.json \
  classical:pixel_nav_runs/classical_seed0/best.pt:pixel_corridor_runs/classical_seed0/best.pt \
  quantum:pixel_nav_runs/quantum_seed0/best.pt:pixel_corridor_runs/quantum_seed0/best.pt
